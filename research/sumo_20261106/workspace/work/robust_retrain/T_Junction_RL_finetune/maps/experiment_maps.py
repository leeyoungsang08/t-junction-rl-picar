from pathlib import Path
import subprocess,xml.etree.ElementTree as ET,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils import binary,sha256
# Coordinates are physically distinct T configurations, same 3-action semantics.
MAPS = {
 "train0":((-240,0),(240,0),(0,-240),(13.89,13.89,13.89)),
 "train1":((-340,-30),(210,20),(25,-170),(13.89,11.11,9.72)),
 "train2":((-195,20),(315,-20),(-20,-190),(11.11,13.89,11.11)),
 "train3":((-400,-75),(225,50),(80,-155),(13.89,11.11,8.33)),
 "train4":((-260,55),(260,-25),(40,-290),(9.72,13.89,11.11)),
 "validation0":((-350,20),(220,-35),(70,-210),(12.5,10.,9.0)),
 "validation1":((-220,-30),(370,35),(-60,-180),(10.,12.5,10.)),
 "test0":((-180,25),(320,-40),(60,-130),(10.,13.0,8.0)),
 "test1":((-360,-55),(200,25),(-70,-260),(13.0,10.,11.)),
}
def build(name,data):
 coords={"J":(0,0),"W":data[0],"E":data[1],"S":data[2]}
 nodes,edges,connections=[ET.Element(n) for n in ("nodes","edges","connections")]
 for d,(x,y) in coords.items(): ET.SubElement(nodes,"node",id=d,x=str(x),y=str(y),type="traffic_light" if d=="J" else "dead_end")
 for i,d in enumerate("WES"):
  for a,b,k in [(d,"J","in"),("J",d,"out")]:
   ET.SubElement(edges,"edge",{"id":f"{d}_{k}","from":a,"to":b,"numLanes":"1","speed":str(data[3][i]),"width":"3.2"})
 for a in "WES":
  for b in "WES":
   if a!=b: ET.SubElement(connections,"connection",{"from":f"{a}_in","to":f"{b}_out","fromLane":"0","toLane":"0"})
 directory=ROOT/"maps/experiment"/name;directory.mkdir(parents=True,exist_ok=True)
 for ext,el in (("nod",nodes),("edg",edges),("con",connections)):
  ET.indent(el);ET.ElementTree(el).write(directory/f"map.{ext}.xml",encoding="utf-8",xml_declaration=True)
 subprocess.run([binary("netconvert"),"-n","map.nod.xml","-e","map.edg.xml","-x","map.con.xml",
     "-o","map.net.xml","--offset.disable-normalization","true","--no-turnarounds","true","--write-license","false"],
     cwd=directory,check=True,stdout=subprocess.DEVNULL)
 path=directory/"map.net.xml"
 tree=ET.parse(path)
 links=[c for c in tree.findall("connection") if c.get("tl")=="J"]
 assert len(links)==6 and {(c.get("from"),c.get("to")) for c in links}=={(f"{a}_in",f"{b}_out") for a in "WES" for b in "WES" if a!=b}
 return sha256(path)
if __name__=="__main__":
 manifest={name:build(name,data) for name,data in MAPS.items()}
 (ROOT/"maps/experiment/manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 print(manifest)
