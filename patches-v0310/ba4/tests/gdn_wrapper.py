"""Complete real GDN module and attention schema; CPU initialization stand-ins only."""
import ast
import importlib.util
import logging
from pathlib import Path
import sys
import types
import torch
import torch._inductor.codecache
import torch._functorch.aot_autograd

HERE = Path(__file__).resolve().parent.parent
source = HERE / 'fixtures/base/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
class Dummy(torch.nn.Module):
    @classmethod
    def register(cls,*a,**kw):return lambda c:c
    @classmethod
    def __class_getitem__(cls,key):return cls

def load_gdn():
    mods={}
    def mod(name):
        if name not in mods:
            m=mods[name]=types.ModuleType(name);m.__path__=[]
            if '.' in name:
                parent,attr=name.rsplit('.',1);setattr(mod(parent),attr,m)
        return mods[name]
    for n in ast.parse(source.read_text()).body:
        if isinstance(n,ast.ImportFrom) and n.module.startswith('vllm'):
            m=mod(n.module)
            for a in n.names:
                if not hasattr(m,a.name):setattr(m,a.name,Dummy)
    platform=types.SimpleNamespace(is_cuda=lambda:True,is_cpu=lambda:False,is_xpu=lambda:False,
        is_rocm=lambda:False,is_device_capability=lambda *a:False,is_device_capability_family=lambda *a:False)
    mod('vllm.platforms').current_platform=platform
    mod('vllm._aiter_ops').rocm_aiter_ops=types.SimpleNamespace(
        are_gdn_triton_kernels_available=lambda:False,is_rdna_gdn_triton_kernels_available=lambda:False)
    mod('vllm.compilation.breakable_cudagraph').eager_break_during_capture=lambda f:f
    mod('vllm.logger').init_logger=logging.getLogger
    u=mod('vllm.utils.torch_utils');u._encode_layer_name=lambda s:s;u.LayerNameType=str
    u.direct_register_custom_op=lambda **kw:None
    einops=types.ModuleType('einops');einops.rearrange=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unused interleaved branch'))
    mods['einops']=einops
    spec=importlib.util.spec_from_file_location('j41r_real_gdn',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    previous={k:sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:spec.loader.exec_module(module)
    finally:
        for k,v in previous.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    return module

observations=[]

@torch.library.custom_op('vllm::qwen_gdn_attention_core',mutates_args=('a_or_z_out','core_attn_out'))
def attention(qkv_or_qkvz:torch.Tensor,b_or_ba:torch.Tensor,a_or_z_out:torch.Tensor,
              core_attn_out:torch.Tensor,layer_name:str,use_aiter:bool=False)->None:
    observations.append((b_or_ba.clone(),a_or_z_out.clone()))
    core_attn_out.copy_((b_or_ba+a_or_z_out).unsqueeze(-1).expand_as(core_attn_out))
@attention.register_fake
def attention_fake(*a,**kw):return None

class Projection(torch.nn.Module):
    def __init__(self,w,h=None,ba=None):
        super().__init__();self.weight=w;self.ba=ba;self.h=h
    def forward(self,x):
        if self.ba is not None:return self.h.apply(self.ba.quant_method,self.ba,x),None
        return torch.nn.functional.linear(x,self.weight),None
class Norm(torch.nn.Module):
    def forward(self,x,z):return x
class Output(torch.nn.Module):
    def forward(self,x):return x,None

def model(h, qkv_kind="fp8"):
    module=load_gdn()
    ba=h.layer(profile=True)
    model=module.QwenGatedDeltaNetAttention.__new__(module.QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(model)
    model.in_proj_ba=Projection(ba.weight,h,ba)
    model.in_proj_qkvz=Projection(torch.nn.Parameter(torch.randn(8192,5120,dtype=torch.bfloat16),requires_grad=False))
    if qkv_kind == "fp8": model.in_proj_qkvz=FP8Projection()
    model.enable_fused_gdn_decode=False;model.gqa_interleaved_layout=False
    model.key_dim=16*128;model.value_dim=48*128;model.num_v_heads=48;model.head_v_dim=128
    model.tp_size=2;model.tp_rank=0;model.disable_tp_for_ba_proj=False;model.prefix='model.layers.0.linear_attn'
    model.norm=Norm();model.out_proj=Output()
    return model, ba

# The run-3 target config and worker log select static tensor FP8 with
# FlashInferFP8ScaledMMLinearKernel for the sibling QKVZ. Execute its real
# apply_weights/apply_scaled_mm and QuantFP8.forward_native (custom_ops=none).
# Only the CUDA bmm leaf and CUDA-only device assertions are substituted.
def function(path, name, cls=None, ns=None, cpu_devices=False):
    import __future__
    tree=ast.parse(path.read_text())
    if cls is not None:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    if cpu_devices:
        node.body=[n for n in node.body if not isinstance(n,ast.Assert) or '.device' not in ast.unparse(n.test)]
    ns={} if ns is None else ns
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec',
                 flags=__future__.annotations.compiler_flag),ns)
    return ns[name]


@torch.library.custom_op('ba4_test::bmm_fp8',mutates_args=())
def bmm_fp8(a:torch.Tensor,b:torch.Tensor,scale_a:torch.Tensor,scale_b:torch.Tensor,
            out_dtype:torch.dtype,backend:str)->torch.Tensor:
    return (torch.bmm(a.to(torch.bfloat16),b.to(torch.bfloat16)) * scale_a * scale_b).to(out_dtype)
@bmm_fp8.register_fake
def bmm_fp8_fake(a,b,scale_a,scale_b,out_dtype,backend):
    return a.new_empty((a.shape[0],a.shape[1],b.shape[2]),dtype=out_dtype)


class FP8Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.randn(8192,5120,dtype=torch.bfloat16).to(torch.float8_e4m3fn).t(),requires_grad=False)
        self.weight_scale=torch.ones(1,dtype=torch.float32)
        self.input_scale=torch.ones(1,dtype=torch.float32)
        self.fp8_dtype=torch.float8_e4m3fn
        self.config=types.SimpleNamespace(out_dtype=torch.bfloat16)
        self.input_quant_key=lambda:None
        self._get_layer_params=lambda layer:(self.weight,self.weight_scale,self.input_scale,None)
        q=types.SimpleNamespace(is_group_quant=False,static=True,group_shape=None,num_token_padding=None)
        fixture=HERE/'fixtures/compile'
        native=function(fixture/'input_quant_fp8.py','forward_native','QuantFP8',dict(
            torch=torch,prep_scale_for_group_broadcast=lambda scale,x,group:scale,
            group_broadcast=lambda scale,k:scale,_FP8_MIN=-448.0,_FP8_MAX=448.0,
            _FP8_DTYPE=torch.float8_e4m3fn))
        self.quant_fp8=lambda x,scale,ub:native(q,x,scale,ub)
        scaled=function(fixture/'flashinfer_utils.py','flashinfer_scaled_fp8_mm',ns=dict(torch=torch,bmm_fp8=bmm_fp8),cpu_devices=True)
        apply_scaled=function(fixture/'fp8_flashinfer.py','apply_scaled_mm','FlashInferFP8ScaledMMLinearKernel',dict(flashinfer_scaled_fp8_mm=scaled))
        self.apply_scaled_mm=lambda **kw:apply_scaled(self,**kw)
        self.apply_weights=function(fixture/'fp8_scaled_mm.py','apply_weights','FP8ScaledMMLinearKernel',dict(torch=torch,as_quantized_activation=lambda *a:None))
    def forward(self,x):
        return self.apply_weights(self,self,x),None
