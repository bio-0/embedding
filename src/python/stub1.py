import json
import os
import io
import sys
import warnings
import traceback
from umap.umap_ import UMAP
import joblib
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore", message=r"\[W033\]", category=UserWarning)

input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


output_json = json.dumps({"status":"started"}, ensure_ascii=False).encode('utf-8')
sys.stdout.buffer.write(output_json)
print()        


def main(input_json):
    output_json = input_json.copy()
    model = SentenceTransformer("nomic-embed",
                                revision='02d96723811f4bb77a80857da07eda78c1549a4d',
                                trust_remote_code=True)
    print("embed loaded")
    with open('umap-models/umap-cluster.joblib', 'rb') as fo:
        reducer = joblib.load(fo)
    with open('umap-models/umap-display.joblib', 'rb') as fo:
        reducer_disp = joblib.load(fo)
    print("umaps loaded")
    vec = model.encode(output_json["text"])
    cluster_vec = reducer.transform(vec)
    print(f'clus vec shape: {cluster_vec.shape}')
    display_vec = reducer_disp.transform(vec)
    print(f'dis vec shape: {display_vec.shape}')
    output_json = {"request": output_json,
                   "response": {"cluster_vec": cluster_vec.tolist(), "display_vec": display_vec.tolist()}}
    return output_json


if __name__=='__main__':
    
    input_json = None
    count = 0
    for line in input_stream:
        
        # read json from stdin
        input_json = json.loads(line)
        
        try:
            # request = main(input_json)
            output = main(input_json)
            # count = count+1
            output = {"request": input_json, "response": output}

        except BaseException as ex:
            ex_type, ex_value, ex_traceback = sys.exc_info()            
            
            output = {"error": ''}           
            output['error'] += "Exception type : %s; \n" % ex_type.__name__
            output['error'] += "Exception message : %s\n" %ex_value
            output['error'] += "Exception traceback : %s\n" %"".join(traceback.TracebackException.from_exception(ex).format())

        output_json = json.dumps(output, ensure_ascii=False).encode('utf-8')
        sys.stdout.buffer.write(output_json)
        print()

        