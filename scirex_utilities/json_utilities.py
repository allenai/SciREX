import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_jsonl(file) :
    return [json.loads(line) for line in open(file )]

def _annotation_to_dict(dc):
    # convenience method
    if isinstance(dc, dict):
        ret = dict()
        for k, v in dc.items():
            k = _annotation_to_dict(k)
            v = _annotation_to_dict(v)
            ret[k] = v
        return ret
    elif isinstance(dc, str):
        return dc
    elif isinstance(dc, (set, frozenset, list, tuple)):
        ret = []
        for x in dc:
            ret.append(_annotation_to_dict(x))
        return tuple(ret)
    else:
        return dc


def annotations_to_jsonl(annotations, output_file, key="doc_id"):
    with open(output_file, "w") as of:
        for ann in sorted(annotations, key=lambda x: x[key]):
            as_json = _annotation_to_dict(ann)
            as_str = json.dumps(as_json, sort_keys=True)
            of.write(as_str)
            of.write("\n")