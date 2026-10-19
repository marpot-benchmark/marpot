import hashlib,json,os,time
from pathlib import Path


def plan_resume(out,identity,resume_own):
    out=Path(out);manifest=out/'run_manifest.json';last=out/'checkpoints/last.ckpt'
    plan={'mode':'fresh','checkpoint':None,'restart_reason':None,'previous_manifest_sha256':None}
    if not manifest.exists():
        if resume_own:raise RuntimeError('cannot resume without this exact job manifest')
        if (out/'checkpoints').exists() and any((out/'checkpoints').iterdir()):
            raise RuntimeError('checkpoint files exist without a run identity; preserve and inspect')
        return plan
    if not resume_own:raise RuntimeError('existing run requires explicit --resume-own')
    previous=json.loads(manifest.read_text())
    changed=[key for key,value in identity.items() if key not in previous or previous[key]!=value]
    if changed:raise RuntimeError('resume identity mismatch: '+', '.join(changed))
    plan['previous_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if last.is_file():plan.update(mode='resume',checkpoint=str(last.resolve()))
    elif last.exists():raise RuntimeError('last.ckpt exists but is not a file')
    else:plan.update(mode='restart_from_initial_checkpoint',
        restart_reason='No owned last.ckpt exists. Restart epoch zero from the identical recorded public/predecessor initialization; preserve previous attempt artifacts.')
    return plan


def begin_attempt(out,identity,plan,attempt=None):
    out=Path(out).resolve();attempt=Path(attempt).resolve() if attempt else out/'attempts'/('manual_'+str(time.time_ns()))
    if not attempt.is_relative_to(out):raise RuntimeError('training attempt must be inside this job directory')
    attempt.mkdir(parents=True,exist_ok=True);receipt=attempt/'training_attempt.json'
    if receipt.exists():raise RuntimeError('training attempt already has a receipt; use a new attempt')
    preserved=None;checkpoints=out/'checkpoints'
    if plan['mode']=='restart_from_initial_checkpoint' and checkpoints.exists() and any(checkpoints.iterdir()):
        preserved=attempt/'preserved_checkpoints';checkpoints.rename(preserved)
    value=dict(plan,identity=identity,started_at=time.time(),preserved_checkpoint_directory=str(preserved) if preserved else None,
               metrics_directory=str(attempt/'metrics/version_0'),previous_metrics_preserved=True)
    temporary=receipt.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,receipt)
    return attempt
