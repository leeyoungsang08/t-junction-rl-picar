import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

work = Path("work/result_audit").resolve()
docs = work.parent / "T_Junction_RL" / "docs"
out = work.parents[1] / "outputs"
corrected = work / "corrected_references"
corrected.mkdir(exist_ok=True)
old = "https://jhc.sjtu.edu.cn/~gjzheng/paper/kdd2018_intellilight/kdd2018_intellilight_paper.pdf"
new = "https://faculty.ist.psu.edu/jessieli/Publications/2018-KDD-IntelliLight.pdf"
with (docs/"paper_review_matrix.csv").open(encoding="utf-8-sig",newline="") as source:
    rows = list(csv.reader(source))
assert sum(any("arxiv.org/abs/2511.00549" in item for item in row) for row in rows) == 1
kept = [[field.replace(old,new) for field in row] for row in rows if not any("arxiv.org/abs/2511.00549" in item for item in row)]
assert len(kept) == 13
with (corrected/"paper_review_matrix.csv").open("w",encoding="utf-8-sig",newline="") as dest:
    csv.writer(dest).writerows(kept)
text = (docs/"literature_review.md").read_text(encoding="utf-8").replace(old,new)
lines = text.splitlines(keepends=True)
idx = next(i for i,line in enumerate(lines) if "Li 등의 지역 제어 프리프린트" in line)
lines[idx] = lines[idx].replace(" Li 등의 지역 제어 프리프린트는 10·20·30% OD 변동 조건을 별도 평가했다.","").replace(", [Li 등 프리프린트](https://arxiv.org/abs/2511.00549)","")
(corrected/"literature_review.md").write_text("".join(lines),encoding="utf-8")
files = [work/name for name in (
    "run_live_audit.py","audit_archived.py","audit_training.py",
    "live_sumo_results.csv","live_summary.json","archived_summary.json","training_summary.json",
    "live_run.log","archived_run.log","training_run.log",
)]
files.extend(p for p in corrected.rglob("*") if p.is_file())
traces = sorted(p for p in (work/"live").rglob("*") if p.is_file())
assert sum(p.name == "tripinfo.xml" for p in traces) == 450
report = out/"T_Junction_RL_Result_Audit_KR.md"
archive = out/"T_Junction_RL_Result_Audit_Evidence.zip"
with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=3) as z:
    z.write(report, report.name)
    for file in files+traces:
        z.writestr(file.relative_to(work).as_posix(),file.read_bytes())
with ZipFile(archive) as z:
    assert z.testzip() is None
    assert sum(n.endswith("/tripinfo.xml") for n in z.namelist()) == 450
print(archive, archive.stat().st_size, "bytes")
