import io
import json
import sys
import traceback
import warnings

import joblib
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", message=r"\[W033\]", category=UserWarning)

input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# output_json = json.dumps({"status": "started"}, ensure_ascii=False).encode('utf-8')
# sys.stdout.buffer.write(output_json)

def main(input_json):

    # Embedding
    vec = model.encode(input_json["text"])

    # Reducing (cluster and display vec)
    cluster_vec = reducer_cluster.transform(vec.reshape(1, -1))
    display_vec = reducer_display.transform(vec.reshape(1, -1))

    return {"cluster_vec": cluster_vec.tolist(), "display_vec": display_vec.tolist()}


if __name__ == '__main__':
    # model = SentenceTransformer("nomic-embed",
    #                            revision='02d96723811f4bb77a80857da07eda78c1549a4d',
    #                            trust_remote_code=True)
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",
                                revision='02d96723811f4bb77a80857da07eda78c1549a4d',
                                trust_remote_code=True)

    # Open umap models
    with open('umap-models/umap-cluster.joblib', 'rb') as fo:
        reducer_cluster = joblib.load(fo)
    with open('umap-models/umap-display.joblib', 'rb') as fo:
        reducer_display = joblib.load(fo)

    input_json = None
    for line in input_stream:

        # read json from stdin
        input_json = json.loads(line)

        try:
            output = {"request": input_json, "response": main(input_json)}

        except BaseException as ex:
            ex_type, ex_value, ex_traceback = sys.exc_info()

            output = {"error": ''}
            output['error'] += "Exception type : %s; \n" % ex_type.__name__
            output['error'] += "Exception message : %s\n" % ex_value
            output['error'] += "Exception traceback : %s\n" % "".join(
                traceback.TracebackException.from_exception(ex).format())

        output_json = json.dumps(output, ensure_ascii=False).encode('utf-8')
        sys.stdout.buffer.write(output_json)
        print()
