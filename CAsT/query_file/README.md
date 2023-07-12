the name means:
- 2019: CAsT-19 dataset. (so on and so forth)
- canon1 or canon0: only consider the previous n canonical passage (n = 1 or 0)
- incontext: to identify if we put the focal query into the context
  - why we do this? if we put the focal query at the end of the context, we provide MRC model an additional information which is "the location of the focal query in terms of the overall context". This may help improve performance.
- the reason why only data2019 is "canon0": CAsT-19 data has no canonical passage provided.