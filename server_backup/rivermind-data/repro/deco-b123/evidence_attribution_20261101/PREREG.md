# 几何竞争约束下的单查询证据读出：首轮冻结方案

2026-11-01，运行结果前冻结。仅检验受约束的候选条件读出，不声称已解决全局归属或方法新颖性。

## 输入与对照

原released NAG与候选pool、原始CLIP、单一query保持固定。最多24个训练视角，排除全部评测图，源视角选择与前轮相同。只用level3互不重叠的SAM mask（至少50像素），避免同视角多尺度重叠mask造成虚假互斥。Gaussian投影/可见性复用已验证实现。固定全部输出Gaussian支撑，不做空间修改。

A：原gateE，208条逐查询重渲染必须与参考完全匹配至1e-6。
G：按几何Hungarian一对一匹配绑定mask到不同root，再以候选的有效匹配观测计算原始文本cos top3。
I：在同一root-mask几何可行边与candidate eligibility内，独立选每视角文本最高mask，再top3。
J：保留不同root之间的mask竞争；对每个candidate，允许选择能够参与近几何最优完整匹配的mask，再取top3。

同root父子候选共享root身份，不作互斥。只在当前released pool的不同root之间竞争，不声称全场景关联。每视角每root至多1个mask，每mask至多1个root，可用dummy不匹配。

## 固定约束

root-mask边：可见Gaussian交集>=20，visible-support IoU>=0.1。geometry objective为所有匹配边的IoU之和，最佳值记G_v。候选mask用于评分还必须满足candidate自身可见支撑与mask支撑的双向覆盖>=0.1、交集>=20。

J中某root-mask边可用，当且仅当强制该边后，其他root排除该mask的最佳Hungarian总分加上强制边IoU >=0.9 G_v（数值tol1e-10）。这是逐视角近最优匹配约束；0.9是预先指定pilot阈值，不根据结果调节。

逐视角约束独立，故可精确计算强制边匹配可行性，再每视角选最高文本分、跨视角top3；等价于该逐视角约束模型中选3个不同视角证据的最大化，不需要MILP。它没有额外的跨视角pairwise身份约束，不能夸大为完整joint inference。

每candidate要求G/I/J均有>=3有效源视角；否则三种新分数共同回退到原id_top3（原NaN沿用旧查询内均值填充）。不删除缺失query。报告共同fallback率、真实改变的token数、mask竞争比例、匹配保留比。不同candidate各自条件下可采用不同匹配解释，不是一次全candidate共享assignment。

## 评价与唯一通过条件

所有新分数分别进入原z(NAG)+z(signal)、released gap与原gateE结构，lambda固定1；另报信号argmax单候选IoU以定位读出效果。GT只在完成分数和选择之后用于IoU/冻结分族，不参与任何匹配或阈值。

主指标按query->image->scene->四scene macro；分族按query均值。2000次按场景分层/图cluster paired bootstrap，seed20261030。J相对A macro>=+.01、CI下界>0、每scene下降不超过.01、found下降不超过.005，并且J优于G和I，才通过。若约束不改变独立读出，判为该约束机制无作用，不补参数sweep。固定pool实验不可能解决池外recall，不作该主张。

所有远端文件、临时目录、新缓存、screen socket放数据盘 `/root/rivermind-data/repro/deco-b123/evidence_attribution_20261101`。
