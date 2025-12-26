from base_ascii_module import SimpleAsciiModule

class SSIMModel(SimpleAsciiModule):
    def __init__(self, chars):
        super(SSIMModel, self).__init__()
        self.chars = chars
        self.char_densities = self.chars.mean(dim=(1, 2, 3))

    def forward(self, img_tensor):
        unfold = nn.Unfold(kernel_size=(H, W), stride=(H, W), padding=0)

        # threshold for those hard edges
        # img_tensor = img_tensor.clamp(0, 1).round()

        # called horse bc that was the og image
        horse = unfold(img_tensor).squeeze(0).T.view(-1, 1, H, W)

        num_tiles = len(horse)
        num_refs = len(self.chars)
        all_scores = torch.zeros(num_tiles, num_refs)

        chunk = horse  # (chunk_size, 1, H, W)
        cs = chunk.shape[0]

        # Expand for broadcasting
        chunk_exp = chunk[:, None].expand(-1, num_refs, -1, -1, -1).reshape(-1, 1, H, W)
        refs_exp = self.chars[None].expand(cs, -1, -1, -1, -1).reshape(-1, 1, H, W)

        scores = kornia.metrics.ssim(chunk_exp, refs_exp, 3).mean(dim=(1, 2, 3))
        all_scores = scores.view(cs, num_refs)

        # Density penalization
        tile_densities = chunk.mean(dim=(1, 2, 3))  # (num_tiles,)
        density_diff = (tile_densities[:, None] - self.char_densities[None, :]).abs()

        # Combine: SSIM rewards shape match, penalty for density mismatch
        all_scores = all_scores - 3 * density_diff  # tune this weight

        # Now build your string
        best_scores, best_indices = all_scores.max(dim=1)
        num_cols = img_tensor.shape[-1] // W

        return best_indices.view(-1, num_cols)