import json
from pathlib import Path
import pytest
from experiments.probe1.analysis.event_detector import Event, first_stable_grasp, handover_timing, seconds_to_samples
from experiments.probe1.analysis.failure_analyzer import classify_episode
from experiments.probe1.analysis.statistics import clustered_delta_cfr, summarize
from experiments.probe1.rollout.recorder import EpisodeRecorder, validate_episode_summary

def row(i, success, failure=None, task="t", run="r", infra=None): return {"episode_id":i,"run_id":run,"task_name":task,"success":success,"primary_failure":failure,"infrastructure_error":infra}
def test_failure_contract_and_outcome_deduplication():
    assert classify_episode(success=True,candidates=[],outcomes=["collision"])["primary_failure"] is None
    got=classify_episode(success=False,candidates=[{"label":"left_execution","time_s":1,"evidence":"x"},{"label":"temporal_coordination","time_s":1,"evidence":"y"}],outcomes=["collision","collision"])
    assert got["primary_failure"] == "other_uncertain" and got["outcomes"] == ["collision"]
def test_grasp_contact_chatter_and_sample_boundary():
    samples=[{"time_s":i/10,"contact":c,"relative_motion_m":.001} for i,c in enumerate([1,1,0,1,1,1])]
    event=first_stable_grasp(samples,sample_hz=10,stable_seconds=.3,max_relative_motion_m=.01)
    assert event.onset_s == .3 and event.confirmed_s == .5 and seconds_to_samples(.21,10)==3
def test_handover_early_and_never_grasped():
    release=Event("release",1.0,1.1,"rule",{})
    assert handover_timing(release,None,early_margin_s=.2)["delta_t_s"] is None
    grasp=Event("grasp",1.5,1.6,"rule",{})
    assert handover_timing(release,grasp,early_margin_s=.2)["early_release_candidate"] is True
def test_metrics_edge_cases_duplicates_and_bootstrap_reproducible():
    assert summarize([])["SR"] is None
    all_fail=summarize([row(1,False,"temporal_coordination")]); assert all_fail["SR"]==0 and all_fail["CFR_fail"]==1
    no_fail=summarize([row(1,True)]); assert no_fail["CFR_fail"] is None
    with pytest.raises(ValueError): summarize([row(1,True),row(1,False)])
    rows=[row(1,False,"temporal_coordination","s"),row(2,True,None,"w")]
    assert clustered_delta_cfr(rows,{"s":"strong","w":"weak"},samples=50,seed=4)==clustered_delta_cfr(rows,{"s":"strong","w":"weak"},samples=50,seed=4)
def test_recorder_null_availability_and_no_overwrite(tmp_path: Path):
    rec=EpisodeRecorder(tmp_path,"run","handover_block",0,7,30,1/250)
    rec.record_step(simulator_step=None,control_step=0,observation={"joint_action":{},"endpose":{}},raw_action_chunk=[[1]],sent_action=[1],action_type="qpos",chunk_id=0,chunk_start_step=0,action_index_in_chunk=0,executed_length=1)
    summary=rec.finish(success=False,termination_reason="timeout")
    step=json.loads((summary.parent/"steps.jsonl").read_text()); assert step["contacts"] is None and not step["availability"]["contacts"]
    with pytest.raises(FileExistsError): EpisodeRecorder(tmp_path,"run","handover_block",0,7,30,1/250)
def test_summary_schema_contract():
    valid={"task_name":"t","episode_id":1,"run_id":"r","eval_seed":2,"success":True,"termination_reason":"success","primary_failure":None,"outcomes":[]}
    validate_episode_summary(valid)
    with pytest.raises(ValueError): validate_episode_summary({**valid,"success":False})
    with pytest.raises(ValueError): validate_episode_summary({**valid,"outcomes":["drop","drop"]})
