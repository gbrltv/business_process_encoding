import os

from pm4py.objects.petri.importer import importer as pnml_importer
from pm4py.visualization.petrinet import visualizer as pn_visualizer

path = 'models/pnml/'
for file in os.listdir(path):
    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(path, file))
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: 'pdf'})

    out_name = 'models/visualization/' + file.split('.')[0] + '.pdf'
    pn_visualizer.save(gviz, out_name)
