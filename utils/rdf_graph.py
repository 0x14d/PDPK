from statistics import quantiles
from pandas import DataFrame
from rdflib import Dataset, Graph, URIRef, Literal, Namespace, RDF, XSD
from rdflib.namespace import DefinedNamespace, Namespace
from iribaker import to_iri

class PPKG_LITERALS(DefinedNamespace):
    startsAt: URIRef
    endsAt: URIRef
    implies: URIRef
    quantifies: URIRef
    quantified_by: URIRef
    relatively_quantified_by: URIRef
    absolutely_quantified_by: URIRef

    _NS = Namespace("http://purl.org/pdpk/literals#")

lit_dict = {
    "starts at": PPKG_LITERALS.startsAt,
    "ends at": PPKG_LITERALS.endsAt,
    "implies": PPKG_LITERALS.implies,
    "quantifies": PPKG_LITERALS.quantifies,
    'quantified by': PPKG_LITERALS.quantified_by,
    'relatively quantified by': PPKG_LITERALS.relatively_quantified_by,
    'absolutely quantified by': PPKG_LITERALS.absolutely_quantified_by
}

class RDFGraph:

    _graph: Graph
    _edges: DataFrame
    _metadata: DataFrame
    _dataset: Dataset

    _data_uri: str = 'http://purl.org/pdpk/resource/'

    def __init__(self, edges: DataFrame, metadata: DataFrame, graph_without_shortcut=True, literal_graph=False) -> None:
        """initializes RDFGraph, runs the graph setup and generates the graph

        Args:
            triples (DataFrame): _description_
            metadata (DataFrame): _description_
            graph_without_shortcut (bool, optional): graph representation with shortcut. Defaults to True.
        """        """"""
        self._edges = edges
        self._metadata = metadata
        self._dataset = Dataset()
        self._literal_graph= literal_graph

        self._rdf_setup()
        self._gen_graph()
        return

    @property
    def graph(self):
        return self._graph

    @property
    def data_uri(self):
        return self._data_uri

    def save_rdf_graph(self, file_name: str):
        """
        saves rdf graph as ttl file
        :param file_name: file name with ending .ttl
        """
        self._dataset.serialize(format='ttl')
        with open(file_name, 'wb') as f:
            self._graph.serialize(f, format='ttl')

    def _rdf_setup(self):
        """configure all nevessary setups for rdf graph generation
        """
        # A namespace for our resources
        data_namespace = Namespace(self._data_uri)

        # A namespace for our vocabulary items (schema information, RDFS, OWL classes and properties etc.)
        vocab_namespace = Namespace('http://purl.org/pdpk/vocab/')

        # The URI for our graph
        graph_uri = URIRef('http://purl.org/pdpk/graph')

        # We initialize a dataset, and bind our namespaces

        self._dataset.bind('ppkgdata', data_namespace)
        self._dataset.bind('ppkgvocab', vocab_namespace)

        # We then get a new graph object with our URI from the dataset.
        self._graph = self._dataset.graph(graph_uri)

    def _gen_graph(self) -> Graph:
        relations = {}
        for _, row in self._edges.iterrows():
            if not self._literal_graph:
                self.add_standard_relation(row, relations)
            else:
                try:
                    lit = row['literal_included']
                except:
                    lit = 'None'

                if lit == 'None':
                    self.add_standard_relation(row, relations)
                elif lit=="From":
                    source_node = Literal(
                        self._metadata.loc[row['from']]['name'], datatype=XSD['float']
                    )
                    target_node = URIRef(
                        to_iri(self._data_uri + str(self._metadata.loc[row['to']]['name']))
                    )

                    self._graph.add(
                        (source_node, lit_dict[row['rel']], target_node)
                    )
                elif lit=="To":
                    target_node = Literal(
                        self._metadata.loc[row['to']]['name'], datatype=XSD['float']
                    )
                    source_node = URIRef(
                        to_iri(self._data_uri + str(self._metadata.loc[row['from']]['name']))
                    )

                    self._graph.add(
                        (source_node, lit_dict[row['rel']], target_node)
                    )
            
    
    def add_standard_relation(self, row, relations):
        source_node = URIRef(
            to_iri(self._data_uri + str(self._metadata.loc[row['from']]["name"]))
        )
        target_node = URIRef(
            to_iri(self._data_uri + str(self._metadata.loc[row['to']]["name"]))
        )
        rel = row['rel']
        if not rel in relations.keys():
            relations[rel] = URIRef(to_iri(self._data_uri + str(rel)))
        rel_ref = relations[rel]
            
        self._graph.add(
            (source_node, rel_ref, target_node)
        )

        self._graph.add((source_node, RDF.type, Literal(
            str(self._metadata.loc[row['from']]["type"]), datatype=XSD['string']
        )))
        self._graph.add((target_node, RDF.type, Literal(
            str(self._metadata.loc[row['to']]["type"]), datatype=XSD['string']
        )))
