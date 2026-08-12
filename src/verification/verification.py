
import pickle
g = pickle.load(open('data/processed/graph_enriched_corrected.pkl','rb'))
print('denso in-degree:', g.in_degree('denso'))
print('denso out-degree:', g.out_degree('denso'))
print('denso total (in+out):', g.in_degree('denso') + g.out_degree('denso'))