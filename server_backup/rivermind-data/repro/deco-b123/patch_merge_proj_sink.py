from pathlib import Path

path = Path('/tmp/merge_proj.py')
text = path.read_text()
start = text.index('    @torch.no_grad()\n    def proj_gaussian_features(')
end = text.index('    def extract_gaussian_features(', start)
new = r'''    @torch.no_grad()
    def proj_gaussian_features(
        self, gaussians, views, feature_level, gau2sp, pt_sp_label, device="cuda"
    ):
        """Original projection plus optional frozen-NAG observation export."""
        from pathlib import Path
        WEIGHT_THRESHOLD = 0.0001
        RATIO_THRESHOLD = 0.3
        sp_uni, sp_gnum = torch.unique(gau2sp, return_counts=True, sorted=True)
        feat_dim = views[0].semantic["sem"].to(device).shape[1]
        sp_feature_list = [[] for _ in range(sp_uni.shape[0])]
        export = getattr(self.args, "observation_out", None)
        obs_features, obs_weights, obs_regions, obs_views = [], [], [], []
        member_offsets, member_ids = [0], []
        visible_offsets, visible_ids = [0], []
        if export:
            members_by_sp = []
            for sp in range(len(sp_feature_list)):
                ids = torch.nonzero(gau2sp == sp, as_tuple=False).flatten().cpu().numpy().astype(np.int64)
                members_by_sp.append(ids)

        pipeline = self.pipeline
        background = self.background
        for view_idx, view in enumerate(tqdm(views, desc="Projection progress")):
            render_pkg = render_point(view, gaussians, pipeline, background)
            weight = render_pkg["weight"]
            gau_mask = weight > WEIGHT_THRESHOLD
            gt_feature = view.semantic["sem"].to(device)
            seg_map = view.semantic["seg_map"][feature_level]
            img_mask = view.semantic["fg_mask"][feature_level]
            seg_min = seg_map[img_mask].min().item()
            seg_max = seg_map[img_mask].max().item()
            # Existing ramen cache stores CLIP maps as float16; the original
            # projection uses float32 assignment ratios. Promote only dtype.
            view_level_feature = gt_feature[seg_min : seg_max + 1].float()
            seg_num = seg_max - seg_min + 2
            sp_mask_mat = self._get_superpoint_mask_ratio(
                gau2sp, pt_sp_label[view_idx], seg_num
            )
            sp_mask_mat[sp_mask_mat < RATIO_THRESHOLD] = 0
            sp_feat = sp_mask_mat @ view_level_feature
            sp_index = gau2sp[gau_mask]
            unique_sp, sp_gau_count = torch.unique(sp_index, return_counts=True)
            for sp_idx, sp in enumerate(unique_sp):
                sp_int = int(sp.item())
                visibility_portion = float(sp_gau_count[sp_idx].item() / sp_gnum[sp].item())
                feat = F.normalize(sp_feat[sp], p=2, dim=-1) * visibility_portion
                if feat.any():
                    sp_feature_list[sp_int].append(feat)
                    if export:
                        obs_features.append((feat / max(visibility_portion, 1e-12)).detach().float().cpu().numpy())
                        obs_weights.append(visibility_portion)
                        obs_regions.append(sp_int)
                        obs_views.append(view_idx)
                        vis = torch.nonzero(gau_mask & (gau2sp == sp), as_tuple=False).flatten().cpu().numpy().astype(np.int64)
                        visible_ids.extend(vis.tolist())
                        visible_offsets.append(len(visible_ids))
                        mem = members_by_sp[sp_int]
                        member_ids.extend(mem.tolist())
                        member_offsets.append(len(member_ids))

        sp_features = []
        for sp in range(len(sp_feature_list)):
            if sp_feature_list[sp]:
                features = torch.stack(sp_feature_list[sp], dim=0)
                sp_features.append(self._filter_feature_outliers(features, device))
            else:
                sp_features.append(torch.zeros(feat_dim, device=device))
        result = F.normalize(torch.stack(sp_features, dim=0), p=2, dim=-1)
        if export:
            out = Path(export)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                features=np.asarray(obs_features, dtype=np.float32),
                weights=np.asarray(obs_weights, dtype=np.float32),
                region_ids=np.asarray(obs_regions, dtype=np.int64),
                view_ids=np.asarray(obs_views, dtype=np.int64),
                visible_gaussian_offsets=np.asarray(visible_offsets, dtype=np.int64),
                visible_gaussian_ids=np.asarray(visible_ids, dtype=np.int64),
                member_gaussian_offsets=np.asarray(member_offsets, dtype=np.int64),
                member_gaussian_ids=np.asarray(member_ids, dtype=np.int64),
            )
        return result

'''
path.write_text(text[:start] + new + text[end:])
