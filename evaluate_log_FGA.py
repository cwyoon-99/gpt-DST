import os
import json
from evaluate.evaluate_FGA import FGA

output_dir = 'expts/230523_0057-gpt35_mw24_1p_v2_full_history_737'

fga_result = FGA(os.path.join(output_dir, "running_log.json"))
print(fga_result)