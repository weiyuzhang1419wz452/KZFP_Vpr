#!/usr/bin/env Rscript
# phylogeny_krab.R
# ─────────────────────────────────────────────────────────────────────────────
# KZFP KRAB-domain phylogenetic tree colored by Vpr interaction (ipTM score).
#
# Inputs  (produced by analyze_krab_vpr.py --only phylogeny/summary):
#   analysis/krab_aligned.fasta   — MAFFT-aligned KRAB sequences
#   analysis/iptm_summary.tsv     — per-gene max ipTM for KRAB and ZNF domains
#
# Outputs → analysis/
#   krab_phylogeny_iptm.pdf / .png   — tree colored by KRAB ipTM
#   krab_phylogeny_krab_vs_znf.pdf   — side-by-side KRAB vs ZNF coloring
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(ape)
  library(seqinr)
  library(ggtree)
  library(treeio)
  library(ggplot2)
  library(ggnewscale)
  library(dplyr)
  library(tidyr)
})

options(ignore.negative.edge = TRUE)   # NJ can produce small negative branch lengths
BASE_DIR <- "/local/workdir/wz452/script/project/KZFP_Vpr"
OUT_DIR  <- file.path(BASE_DIR, "analysis")

# ── 1. Load ipTM scores ───────────────────────────────────────────────────────
iptm <- read.table(
  file.path(OUT_DIR, "iptm_summary.tsv"),
  header = TRUE, sep = "\t", stringsAsFactors = FALSE
)
# Make safe IDs matching FASTA headers (hyphens → underscores)
iptm$safe_id <- gsub("[-.]", "_", iptm$gene)

cat(sprintf("Loaded %d genes with ipTM scores\n", nrow(iptm)))
cat(sprintf("KRAB ipTM ≥ 0.5: %d genes\n", sum(iptm$krab_iptm >= 0.5, na.rm = TRUE)))

# ── 2. Read alignment and build NJ tree ──────────────────────────────────────
aln_seqinr <- read.alignment(file.path(OUT_DIR, "krab_aligned.fasta"), format = "fasta")
cat(sprintf("Alignment: %d sequences x %d positions\n",
            aln_seqinr$nb, nchar(aln_seqinr$seq[[1]])))

# Compute identity-based distance matrix (seqinr handles large AA alignments)
dist_mat <- dist.alignment(aln_seqinr, matrix = "identity")
tree     <- nj(dist_mat)       # neighbor-joining
# midpoint rooting: find the two most distant tips and root between them
tree     <- root(tree, outgroup = which.max(cophenetic(tree)[1,]), resolve.root = TRUE)

cat(sprintf("Tree: %d tips\n", length(tree$tip.label)))

# ── 3. Compute per-node mean ipTM (tips + internal nodes) ────────────────────
# Internal node value = mean ipTM of all descendant tips.
# This lets us color branches by the interaction strength of the clade below.
node_iptm_df <- function(tree, tip_iptm_named) {
  n_tips  <- length(tree$tip.label)
  n_nodes <- tree$Nnode
  n_total <- n_tips + n_nodes

  # Collect descendant tip indices for every node
  desc <- vector("list", n_total)
  for (i in seq_len(n_tips)) desc[[i]] <- i
  # Traverse edges from leaves toward root (reverse post-order)
  for (i in rev(seq_len(nrow(tree$edge)))) {
    parent <- tree$edge[i, 1]
    child  <- tree$edge[i, 2]
    desc[[parent]] <- c(desc[[parent]], desc[[child]])
  }
  # Mean ipTM over all descendant tips
  vals <- sapply(seq_len(n_total), function(nd) {
    tip_idx <- desc[[nd]][desc[[nd]] <= n_tips]
    if (length(tip_idx) == 0) return(NA_real_)
    mean(tip_iptm_named[tree$tip.label[tip_idx]], na.rm = TRUE)
  })
  tibble(node = seq_len(n_total), node_iptm = vals)
}

# ── 4. Attach ipTM scores to tip labels ──────────────────────────────────────
tip_df <- tibble(label = tree$tip.label) %>%
  left_join(
    iptm %>% select(label = safe_id, krab_iptm, znf_iptm),
    by = "label"
  ) %>%
  mutate(
    krab_iptm   = replace_na(krab_iptm, 0),
    znf_iptm    = replace_na(znf_iptm,  0),
    interaction = case_when(
      krab_iptm >= 0.6 ~ "Strong (>=0.6)",
      krab_iptm >= 0.5 ~ "Moderate (0.5-0.6)",
      krab_iptm >= 0.3 ~ "Weak (0.3-0.5)",
      TRUE             ~ "None (<0.3)"
    ),
    interaction = factor(interaction,
                         levels = c("Strong (>=0.6)", "Moderate (0.5-0.6)",
                                    "Weak (0.3-0.5)", "None (<0.3)"))
  )

# Build node-level data (tips + internal) for branch coloring
tip_iptm_vec <- setNames(tip_df$krab_iptm, tip_df$label)
nd_df        <- node_iptm_df(tree, tip_iptm_vec)

# Shared color scale
iptm_colors <- c("white", "#fee8c8", "#fdbb84", "#e34a33", "#7a0000")
iptm_scale_color <- scale_color_gradientn(
  colors = iptm_colors, limits = c(0, 1),
  name = "max ipTM\nwith Vpr",
  guide = guide_colorbar(barwidth = 0.8, barheight = 6)
)

# ── 5. Plot 1: continuous ipTM — branches + labels colored ───────────────────
p1 <- ggtree(tree, layout = "rectangular", size = 0.15) %<+% tip_df %<+% nd_df +
  geom_tree(aes(color = node_iptm), linewidth = 0.4) +
  geom_tippoint(aes(color = krab_iptm), size = 0.9) +
  geom_tiplab(aes(color = krab_iptm), size = 1.4, align = FALSE, offset = 0.002) +
  iptm_scale_color +
  labs(title = "KZFP KRAB-domain phylogeny",
       subtitle = "Branch and label color = max ipTM with Vpr (darker = stronger interaction)") +
  theme_tree2() +
  theme(
    plot.title    = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 8,  color = "grey30"),
    legend.position = "right",
    legend.title    = element_text(size = 8),
    legend.text     = element_text(size = 7),
  )

ggsave(file.path(OUT_DIR, "krab_phylogeny_iptm.pdf"),
       p1, width = 14, height = max(12, length(tree$tip.label) * 0.18),
       limitsize = FALSE)
ggsave(file.path(OUT_DIR, "krab_phylogeny_iptm.png"),
       p1, width = 14, height = max(12, length(tree$tip.label) * 0.18),
       dpi = 150, limitsize = FALSE)
cat("Saved: krab_phylogeny_iptm.pdf / .png\n")

# ── 5. Plot 2: discrete interaction category ──────────────────────────────────
cat_colors <- c(
  "Strong (>=0.6)"      = "#7a0000",
  "Moderate (0.5-0.6)" = "#e34a33",
  "Weak (0.3-0.5)"     = "#fdbb84",
  "None (<0.3)"        = "#d9d9d9"
)

p2 <- ggtree(tree, layout = "rectangular", size = 0.15, color = "grey40") %<+% tip_df +
  geom_tippoint(aes(color = interaction), size = 1.2) +
  geom_tiplab(aes(color = interaction), size = 1.4, align = FALSE, offset = 0.002) +
  scale_color_manual(values = cat_colors, name = "Vpr interaction") +
  labs(title = "KZFP KRAB-domain phylogeny",
       subtitle = "Leaf color = predicted Vpr interaction strength") +
  theme_tree2() +
  theme(
    plot.title    = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 8,  color = "grey30"),
    legend.position = "right",
    legend.title    = element_text(size = 9),
    legend.text     = element_text(size = 8),
  )

ggsave(file.path(OUT_DIR, "krab_phylogeny_categories.pdf"),
       p2, width = 14, height = max(12, length(tree$tip.label) * 0.18),
       limitsize = FALSE)
ggsave(file.path(OUT_DIR, "krab_phylogeny_categories.png"),
       p2, width = 14, height = max(12, length(tree$tip.label) * 0.18),
       dpi = 150, limitsize = FALSE)
cat("Saved: krab_phylogeny_categories.pdf / .png\n")


# ── 7. Summary stats by tree clade ───────────────────────────────────────────
cat("\n── ipTM summary ─────────────────────────────────────────\n")
cat(sprintf("%-25s %5s %5s\n", "Category", "n", "pct"))
iptm %>%
  mutate(cat = case_when(
    krab_iptm >= 0.6 ~ "Strong KRAB (≥0.6)",
    krab_iptm >= 0.5 ~ "Moderate KRAB (0.5–0.6)",
    krab_iptm >= 0.3 ~ "Weak KRAB (0.3–0.5)",
    TRUE             ~ "None (<0.3)"
  )) %>%
  count(cat) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(desc(n)) %>%
  { cat(sprintf("%-25s %5d %4.1f%%\n", .$cat, .$n, .$pct)) ; . }

cat("\nTop 20 KRAB interactors:\n")
iptm %>%
  arrange(desc(krab_iptm)) %>%
  head(20) %>%
  mutate(znf_iptm = ifelse(is.na(znf_iptm), "N/A", sprintf("%.3f", znf_iptm))) %>%
  { cat(sprintf("  %-15s  KRAB=%.3f  ZNF=%s\n", .$gene, .$krab_iptm, .$znf_iptm)) ; . }

# ── 8. Circular (fan) tree — continuous ipTM ─────────────────────────────────
p_circ <- ggtree(tree, layout = "circular", size = 0.15) %<+% tip_df %<+% nd_df +
  geom_tree(aes(color = node_iptm), linewidth = 0.4) +
  geom_tippoint(aes(color = krab_iptm), size = 0.8) +
  geom_tiplab(aes(color = krab_iptm), size = 1.2, offset = 0.005) +
  iptm_scale_color +
  labs(title    = "KZFP KRAB-domain phylogeny (circular)",
       subtitle = "Branch and label color = max ipTM with Vpr") +
  theme_tree2() +
  theme(
    plot.title       = element_text(size = 11, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 8,  color = "grey30", hjust = 0.5),
    legend.position  = "right",
    legend.title     = element_text(size = 8),
    legend.text      = element_text(size = 7),
  )

ggsave(file.path(OUT_DIR, "krab_phylogeny_circular.pdf"),
       p_circ, width = 18, height = 18, limitsize = FALSE)
ggsave(file.path(OUT_DIR, "krab_phylogeny_circular.png"),
       p_circ, width = 18, height = 18, dpi = 150, limitsize = FALSE)
cat("Saved: krab_phylogeny_circular.pdf / .png\n")

# ── 9. Circular tree — discrete categories ───────────────────────────────────
p_circ_cat <- ggtree(tree, layout = "circular", size = 0.15) %<+% tip_df %<+% nd_df +
  geom_tree(aes(color = node_iptm), linewidth = 0.4) +
  scale_color_gradientn(colors = iptm_colors, limits = c(0, 1), guide = "none") +
  new_scale_color() +
  geom_tippoint(aes(color = interaction), size = 1.1) +
  geom_tiplab(aes(color = interaction), size = 1.2, offset = 0.005) +
  scale_color_manual(values = cat_colors, name = "Vpr interaction") +
  labs(title    = "KZFP KRAB-domain phylogeny (circular)",
       subtitle = "Color = predicted Vpr interaction strength") +
  theme_tree2() +
  theme(
    plot.title       = element_text(size = 11, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 8,  color = "grey30", hjust = 0.5),
    legend.position  = "right",
    legend.title     = element_text(size = 9),
    legend.text      = element_text(size = 8),
  )

ggsave(file.path(OUT_DIR, "krab_phylogeny_circular_categories.pdf"),
       p_circ_cat, width = 18, height = 18, limitsize = FALSE)
ggsave(file.path(OUT_DIR, "krab_phylogeny_circular_categories.png"),
       p_circ_cat, width = 18, height = 18, dpi = 150, limitsize = FALSE)
cat("Saved: krab_phylogeny_circular_categories.pdf / .png\n")

# ── 10. Circular tree: KRAB color + ZNF size dual-encoding ───────────────────
# Color = KRAB ipTM (red gradient); point size = ZNF ipTM (bigger = higher ZNF interaction)
p_circ_kz <- ggtree(tree, layout = "circular", linewidth = 0.15) %<+% tip_df %<+% nd_df +
  geom_tree(aes(color = node_iptm), linewidth = 0.4) +
  geom_tippoint(aes(color = krab_iptm, size = znf_iptm + 0.05), shape = 16, alpha = 0.9) +
  geom_tiplab(aes(color = krab_iptm), size = 1.1, offset = 0.005) +
  scale_color_gradientn(
    colors = c("white", "#fee8c8", "#fdbb84", "#e34a33", "#7a0000"),
    limits = c(0, 1), name = "KRAB ipTM\n(color)"
  ) +
  scale_size_continuous(range = c(0.4, 3.0), name = "ZNF ipTM\n(size)") +
  labs(title    = "KZFP KRAB-domain phylogeny (circular)",
       subtitle = "Color = KRAB ipTM with Vpr;  point size = ZNF ipTM with Vpr") +
  theme_tree2() +
  theme(
    plot.title    = element_text(size = 11, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 8,  color = "grey30", hjust = 0.5),
    legend.position = "right"
  )

ggsave(file.path(OUT_DIR, "krab_phylogeny_circular_krab_znf.pdf"),
       p_circ_kz, width = 20, height = 20, limitsize = FALSE)
ggsave(file.path(OUT_DIR, "krab_phylogeny_circular_krab_znf.png"),
       p_circ_kz, width = 20, height = 20, dpi = 150, limitsize = FALSE)
cat("Saved: krab_phylogeny_circular_krab_znf.pdf / .png\n")

# ── 11. KRAB-B-only phylogeny ─────────────────────────────────────────────────
cat("\n── Building KRAB-B phylogeny ────────────────────────────────────────────\n")

krab_b_aln_path <- file.path(OUT_DIR, "krab_b_aligned.fasta")
if (!file.exists(krab_b_aln_path)) {
  cat("krab_b_aligned.fasta not found — skipping KRAB-B phylogeny\n")
} else {
  aln_b       <- read.alignment(krab_b_aln_path, format = "fasta")
  cat(sprintf("KRAB-B alignment: %d sequences x %d positions\n",
              aln_b$nb, nchar(aln_b$seq[[1]])))

  dist_b  <- dist.alignment(aln_b, matrix = "identity")
  # njs() handles missing values that arise from identical/gapped sequences
  tree_b  <- njs(dist_b)
  tree_b  <- root(tree_b, outgroup = which.max(cophenetic(tree_b)[1,]),
                  resolve.root = TRUE)
  cat(sprintf("KRAB-B tree: %d tips\n", length(tree_b$tip.label)))

  # Attach ipTM scores to tips (safe_id column must match FASTA header names)
  tip_b_df <- tibble(label = tree_b$tip.label) %>%
    left_join(
      iptm %>% select(label = safe_id, krab_iptm, znf_iptm),
      by = "label"
    ) %>%
    mutate(
      krab_iptm   = replace_na(krab_iptm, 0),
      znf_iptm    = replace_na(znf_iptm,  0),
      interaction = case_when(
        krab_iptm >= 0.6 ~ "Strong (>=0.6)",
        krab_iptm >= 0.5 ~ "Moderate (0.5-0.6)",
        krab_iptm >= 0.3 ~ "Weak (0.3-0.5)",
        TRUE             ~ "None (<0.3)"
      ),
      interaction = factor(interaction,
                           levels = c("Strong (>=0.6)", "Moderate (0.5-0.6)",
                                      "Weak (0.3-0.5)", "None (<0.3)"))
    )

  tip_iptm_b     <- setNames(tip_b_df$krab_iptm, tip_b_df$label)
  nd_b_df        <- node_iptm_df(tree_b, tip_iptm_b)

  # Circular continuous
  p_b_circ <- ggtree(tree_b, layout = "circular", size = 0.15) %<+% tip_b_df %<+% nd_b_df +
    geom_tree(aes(color = node_iptm), linewidth = 0.4) +
    geom_tippoint(aes(color = krab_iptm), size = 0.8) +
    geom_tiplab(aes(color = krab_iptm), size = 1.2, offset = 0.005) +
    iptm_scale_color +
    labs(title    = "KZFP KRAB-B phylogeny (circular)",
         subtitle = "Branch and label color = max ipTM with Vpr (full-KRAB predictions)") +
    theme_tree2() +
    theme(
      plot.title    = element_text(size = 11, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 8,  color = "grey30", hjust = 0.5),
      legend.position = "right",
      legend.title  = element_text(size = 8),
      legend.text   = element_text(size = 7),
    )

  ggsave(file.path(OUT_DIR, "krab_b_phylogeny_circular.pdf"),
         p_b_circ, width = 18, height = 18, limitsize = FALSE)
  ggsave(file.path(OUT_DIR, "krab_b_phylogeny_circular.png"),
         p_b_circ, width = 18, height = 18, dpi = 150, limitsize = FALSE)
  cat("Saved: krab_b_phylogeny_circular.pdf / .png\n")

  # Circular categorical
  p_b_cat <- ggtree(tree_b, layout = "circular", size = 0.15) %<+% tip_b_df %<+% nd_b_df +
    geom_tree(aes(color = node_iptm), linewidth = 0.4) +
    scale_color_gradientn(colors = iptm_colors, limits = c(0, 1), guide = "none") +
    new_scale_color() +
    geom_tippoint(aes(color = interaction), size = 1.1) +
    geom_tiplab(aes(color = interaction), size = 1.2, offset = 0.005) +
    scale_color_manual(values = cat_colors, name = "Vpr interaction") +
    labs(title    = "KZFP KRAB-B phylogeny (circular)",
         subtitle = "Color = predicted Vpr interaction strength") +
    theme_tree2() +
    theme(
      plot.title    = element_text(size = 11, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 8,  color = "grey30", hjust = 0.5),
      legend.position = "right",
      legend.title  = element_text(size = 9),
      legend.text   = element_text(size = 8),
    )

  ggsave(file.path(OUT_DIR, "krab_b_phylogeny_circular_categories.pdf"),
         p_b_cat, width = 18, height = 18, limitsize = FALSE)
  ggsave(file.path(OUT_DIR, "krab_b_phylogeny_circular_categories.png"),
         p_b_cat, width = 18, height = 18, dpi = 150, limitsize = FALSE)
  cat("Saved: krab_b_phylogeny_circular_categories.pdf / .png\n")
}

cat("\nDone. Outputs in:", OUT_DIR, "\n")
