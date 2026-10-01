# De-identified statistics

Produced with `scripts/per_class_stats.py` on the **auxiliary glyph set** described in
Section 4.1 of the manuscript — externally sourced character images collected to enlarge
the thin tail of the inventory. It is *not* the shared-task training set, whose licence does
not permit redistribution.

| File | Contents |
|---|---|
| `glyph_distribution_summary.txt` | Categories, image count, mean / median / min / max / decile and quartile images per category, share of categories below 5, 10 and 20 images, share of images held by the top 10% and top 1% of categories |
| `glyph_distribution_histogram.csv` | The same distribution as a binned histogram: `images_per_category_bin, categories, share_of_categories_pct` |

The distribution is strongly skewed — median 85 images per category against a mean of 174.9,
with 10.1% of categories holding fewer than 10 images and the top 10% of categories holding
44.2% of all images. This is the shape the manuscript cites when it argues that per-category
performance cannot be read off a single aggregate accuracy figure.

Reproduce with:

```bash
python ../scripts/per_class_stats.py train --jsonl metadata.jsonl --key char --out glyph_distribution
```

No images, character forms or instance-level labels appear in these files. They are released
under CC BY 4.0.
