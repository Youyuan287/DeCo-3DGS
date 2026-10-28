"""Boundary contract for a *production* ReLaGS observer. This is called INSIDE
merge_proj.py where per-region per-view feature & visibility_portion exist.
Not a reconstruction script and not a substitute for production code.
"""
import numpy as np

class ProductionObservationSink:
    def __init__(self):
        self.features=[];self.weights=[];self.regions=[];self.views=[];self.levels=[]
        self.member_ids=[];self.visible_ids=[]
    def add(self, region_id, view_id, sp_feat, visibility_portion,
            member_gaussian_ids, visible_gaussian_ids, level_id):
        x=np.asarray(sp_feat,dtype=np.float32).reshape(-1)
        w=float(visibility_portion)
        members=np.asarray(member_gaussian_ids,dtype=np.int64).reshape(-1)
        visible=np.asarray(visible_gaussian_ids,dtype=np.int64).reshape(-1)
        if not np.isfinite(x).all() or not np.isfinite(w) or w<=0:return
        if np.linalg.norm(x)==0:return
        if not len(members) or len(visible)>len(members):
            raise ValueError("invalid 3D support for production observation")
        self.features.append(x);self.weights.append(w)
        self.regions.append(int(region_id));self.views.append(int(view_id))
        self.levels.append(int(level_id));self.member_ids.append(members);self.visible_ids.append(visible)
    def save(self,path):
        if not self.features: raise ValueError('No production observations captured')
        def pack(rows):
            offsets=np.zeros(len(rows)+1,dtype=np.int64)
            offsets[1:]=np.cumsum([len(x) for x in rows],dtype=np.int64)
            return offsets,np.concatenate(rows) if offsets[-1] else np.empty(0,dtype=np.int64)
        member_offsets,member_values=pack(self.member_ids)
        visible_offsets,visible_values=pack(self.visible_ids)
        np.savez_compressed(path,features=np.stack(self.features),
            weights=np.asarray(self.weights,dtype=np.float32),
            region_ids=np.asarray(self.regions,dtype=np.int64),
            view_ids=np.asarray(self.views,dtype=np.int64),
            level_ids=np.asarray(self.levels,dtype=np.int64),
            member_gaussian_offsets=member_offsets,
            member_gaussian_ids=member_values,
            visible_gaussian_offsets=visible_offsets,
            visible_gaussian_ids=visible_values)
