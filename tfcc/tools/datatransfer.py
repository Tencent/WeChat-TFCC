import argparse
import tensorflow as tf
import numpy as np
import struct

def data_write_to_file(variables, path):
    npzdict = {}
    for var in variables:
        varname = var['name']
        if varname[-2:] == ':0':
            varname = varname[:-2]

        # evaluate values
        value = np.asarray(var['data'])
        value = np.ascontiguousarray(value)
        npzdict[varname] = value

        print(" - dump `%s` shape `%s` ok" % (varname, str(value.shape)))
    np.savez(path, **npzdict)

def get_all_tensor(sess):
    supportDtypes = [
        tf.float32.as_datatype_enum(),
        tf.float64.as_datatype_enum(),
        tf.int8.as_datatype_enum(),
        tf.uint8.as_datatype_enum(),
        tf.int16.as_datatype_enum(),
        tf.uint16.as_datatype_enum(),
        tf.int32.as_datatype_enum(),
        tf.uint32.as_datatype_enum(),
    ]
    result = set()
    for n in sess.graph.as_graph_def().node:
        if n.op != 'Const' or 'value' not in n.attr or not n.attr['value'].tensor.dtype in supportDtypes:
            continue
        result.add(n.name + ':0')
    result.update([ v.name for v in tf.global_variables() ])
    return result

def dump_to_npz(sess, path):
    names = get_all_tensor(sess)
    variables = []
    for n in names:
        try:
            t = sess.graph.get_tensor_by_name(n)
            data = sess.run(t)
            variables.append({'name' : n, 'data': data})
        except:
            print('tensor: [%s] run error, skip.' % n)
    print('----------------------')
    data_write_to_file(variables, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Full path to pb file')
    parser.add_argument('--out', type=str, required=True, help='Full path to output file path')
    args = parser.parse_args()

    with tf.Session() as sess:
        with tf.gfile.FastGFile(args.checkpoint, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        dump_to_npz(sess, args.out)

if __name__ == '__main__':
    main()
