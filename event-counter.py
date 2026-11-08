import utils
import numpy as np
ev = utils.get_event_parameters("/data/user/jstowers/icetop-cnn/simdata/snow2018", composition="phof")
count = ev["energy"].shape[0]

ecomp = {int(c): int((ev["comp"]==c).sum()) for c in np.unique(ev["comp"])}

print(count)
print(ecomp)


