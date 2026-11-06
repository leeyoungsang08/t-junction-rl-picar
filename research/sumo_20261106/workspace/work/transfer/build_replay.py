import json
from pathlib import Path
import xml.etree.ElementTree as ET
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs"
root = ET.parse(HERE/"maps/skew_t.net.xml")
edges={e.get("id"):e for e in root.findall("edge")}
def shape(e):
    return [list(map(float,p.split(","))) for p in e.find("lane").get("shape").split()]
lanes = [shape(e) for k,e in edges.items() if not k.startswith(":")]
lights=[]; anchors=[]
for d in "WES":
    end, stop = shape(edges[f"{d}_in"])
    lights.append([round(a+.045*(b-a),2) for a,b in zip(stop,end)])
    anchors.append([end[0],end[1]+(23 if d!="S" else -20)])
controllers=[]
for name,label,short in [("direct_rainbow_lite_seed42","Direct Rainbow-lite","Direct"),
                         ("queue_proxy","대기열 기준","대기열"),("fixed20","고정식 20초","고정식")]:
    path=HERE/"runs/new/reversal/81000"/name/"trajectory.jsonl"
    frames=[]
    for row in path.read_text().splitlines():
        x=json.loads(row)
        assert x["time"]==len(frames)+1
        frames.append([x["phase"],["green","yellow","all_red"].index(x["stage"]),x["queues"],x["arrived"],
                       [[round(v["x"],1),round(v["y"],1),"WES".index(v["origin"]),int(v["speed"]<.1)]
                        for v in x["vehicles"]]])
    assert len(frames)==1200
    controllers.append(dict(label=label,short=short,frames=frames))
data=json.dumps(dict(lanes=lanes,lights=lights,anchors=anchors,controllers=controllers),ensure_ascii=False,separators=(",",":"))
template=(HERE/"replay_template.html").read_text(encoding="utf-8-sig")
(OUT/"T_Junction_Transfer_Signal_Replay.html").write_text(template.replace("__DATA__",data),encoding="utf-8")
