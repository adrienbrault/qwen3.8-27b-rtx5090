"""The v3 protocol affects preparation/admission only, using the v2 route."""
import copy
import os
import unittest
from unittest import mock


def v3_report():
    from test_cpu import valid_report
    from vllm_ba4 import policy
    from vllm_ba4.evidence import digest
    r=valid_report();r.update(protocol=policy.PROFILE_PROTOCOL,verdict='PROFILE_QUALIFIED')
    binding=r['serving_evidence']['binding'];r['serving_evidence']={'binding':binding}
    inputs={f'provisional-rank{x}.json':str(x)*64 for x in (0,1)}
    pairs=[]
    for n in (0,1):
        manifest={'last.txt':str(n)*64}
        worker=dict(pid='100',kernel=r['selected'],prefixes=sorted(e['prefix'] for e in r['layers']))
        t=dict(control_us=2000,candidate_us=1500,saving_us=500)
        pairs.append(dict(result_dir=f'/synthetic/unit{n}',unit_identity=digest(n),order=['A_S_L1a','L1a_BA4'],
            provisional_inputs=inputs,binding=binding,gates='PASS',aot_cache_decimal_nonce_dirs=0,
            workers={'0':worker,'1':dict(worker,pid='101')},unprofiled=t,profiled={str(rank):dict(t,control_trace_sha256=digest([n,rank,'control']),candidate_trace_sha256=digest([n,rank,'candidate'])) for rank in (0,1)},
            manifest=manifest,manifest_sha256=digest(manifest)))
    r['profile_evidence']=dict(protocol=policy.PROFILE_PROTOCOL,provisional_inputs=inputs,pairs=pairs,
        migration=dict(source_code_sha256='code',target_code_sha256='code',
            source_image_id=binding['candidate_image_id'],target_image_id=binding['candidate_image_id']),
        diagnostic_sha256=digest({k:v for k,v in r.items() if k!='serving_evidence'}))
    b=dict(protocol=policy.PROFILE_PROTOCOL,binding=binding,
        fingerprints=[r['fingerprint'],dict(r['fingerprint'],device_uuid='gpu1')],
        sources=[{k:p[k] for k in ('result_dir','unit_identity','manifest_sha256','manifest')} for p in pairs],
        provisional_inputs=inputs)
    # Existing CPU test fingerprint has no UUID; add actual distinct identities.
    r['fingerprint']=dict(r['fingerprint'],device_uuid='gpu0');b['fingerprints'][0]=r['fingerprint']
    r['profile_evidence']['diagnostic_sha256']=digest({k:v for k,v in r.items() if k not in ('profile_evidence','serving_evidence')})
    return r,b


class V3AdmissionTests(unittest.TestCase):
    def setUp(self):
        from test_cpu import RuntimeTests
        RuntimeTests.setUp(self)

    def prepare(self,r,b):
        with mock.patch.dict(os.environ, VLLM_BATCH_INVARIANT='0',VLLM_SM12X_BA_M4_PROFILE_ONLY='0',
             VLLM_SM12X_BA_M4_RESULT='rank{rank}.json',VLLM_SM12X_BA_M4_BINDING='binding.json'), \
             mock.patch.object(self.runtime,'_read_report',side_effect=lambda path:b if path=='binding.json' else r), \
             mock.patch.object(self.runtime,'fingerprint',return_value=r['fingerprint']), \
             mock.patch.object(self.runtime,'code_sha256',return_value='code'), \
             mock.patch.object(self.runtime,'tensor_sha256',return_value='a'*64):
            self.runtime.prepare_layer(self.layer)

    def test_v3_active_admission_uses_same_selected_kernel_and_live_route(self):
        from test_cpu import Tensor
        r,b=v3_report();self.prepare(r,b)
        state=self.layer._sm12x_ba4
        self.assertIsNotNone(state);self.assertFalse(state.profile_only)
        self.assertEqual(state.name,'torch_mm')
        self.kernels.run_out.assert_called_once()
        self.assertIs(self.runtime.try_apply(self.layer,Tensor((4,5120)),None),state.out)
        self.assertEqual(self.kernels.run_out.call_count,2)

    def test_v3_binding_and_weight_refusals_allocate_nothing(self):
        for mutation in ('schema','source','weight','code','benchmark','fingerprint'):
            r,b=v3_report()
            if mutation=='schema': b['protocol']='old'
            elif mutation=='source': b['sources'][0]['manifest_sha256']='f'*64
            elif mutation=='weight': r['layers'][0]['weight_sha256']='b'*64
            elif mutation=='code': r['code_sha256']='changed'
            elif mutation=='benchmark': r['benchmark_sha256']='changed'
            else: b['fingerprints'][0]={}
            self.prepare(r,b);self.assertIsNone(self.layer._sm12x_ba4,mutation)
        self.torch.empty.assert_not_called();self.kernels.run_out.assert_not_called()

    def test_v2_win_and_profile_only_regression(self):
        from test_cpu import valid_report
        from vllm_ba4 import policy
        r=valid_report()
        self.assertIsNone(policy.admission_reason(r,r['fingerprint'],'code',0))
        r['verdict']='PROVISIONAL';r.pop('serving_evidence')
        self.assertIsNone(policy.admission_reason(r,r['fingerprint'],'code',0,profile_only=True))
        self.assertIsNotNone(policy.admission_reason(r,r['fingerprint'],'code',0))


if __name__=='__main__':unittest.main()
