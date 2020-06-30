import os
from tqdm import tqdm
from pm4py.objects.petri.importer import importer as pnml_importer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from sort_alphanumeric import sort_alphanumeric

path = './models/pnml/'
out_path = './models/visualization/'
for file in tqdm(sort_alphanumeric(os.listdir(path))):
    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(path, file))
    gviz = pn_visualizer.apply(
        net,
        initial_marking,
        final_marking,
        parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: 'eps'})

    out_name = out_path + file.split('.')[0] + '.pdf'
    pn_visualizer.save(gviz, out_name)
