"""
This module provides the class `SyntheticDataGenerator`.
"""

# pylint: disable=too-many-instance-attributes

from copy import deepcopy
from datetime import datetime
from typing import List, Optional, Union
import uuid

import igraph
import rdflib
from rdflib import DCTERMS, RDF, VOID, XSD, Literal, Namespace, URIRef

from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.modules.experiment_generators. \
    abstract_experiment_generator import ExperimentGenerator
from data_provider.knowledge_graphs.generators.abstract_knowledge_graph_generator \
    import KnowledgeGraphGenerator
from data_provider.synthetic_data_generation.modules.noise_generators. \
    abstract_noise_generator import NoiseGenerator
from data_provider.synthetic_data_generation.modules.pq_function_generators. \
    abstract_pq_function_generator import PQFunctionGenerator
from data_provider.synthetic_data_generation.modules.pq_tuple_generators.\
    abstract_pq_tuple_generator import PQTupleGenerator
from data_provider.synthetic_data_generation.types.experiments import GeneratedDataset
from data_provider.synthetic_data_generation.types.generator_arguments  \
    import ExperimentGeneratorArguments, KnowledgeGraphGeneratorArguments, \
        NoiseGeneratorArguments, PQTupleGeneratorArguments
from data_provider.synthetic_data_generation.types.generator_arguments2 \
    import PQFunctionGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples
from data_provider.knowledge_graphs.pq_relation import PQ_Relation


class SyntheticDataGenerator:
    """
    ...
    """

    config: SdgConfig
    """Configuration of the synthetic data generator"""

    pq_functions: GeneratedPQFunctions
    """Generated pq-functions"""

    pq_tuples: GeneratedPQTuples
    """Generated pq-tuples (correlating pq-tuples, expert knowledge etc.)"""

    pq_relations: List[PQ_Relation]
    """Generated pq-relations"""

    _dataset_generator: ExperimentGenerator
    _knowledge_graph_generator: KnowledgeGraphGenerator
    _noise_generator: Optional[NoiseGenerator]
    _pq_function_generator: PQFunctionGenerator
    _pq_tuple_generator: PQTupleGenerator

    _dataset: GeneratedDataset
    _knowledge_graph: igraph.Graph
    _id: str

    def __init__(self, config: Union[SdgConfig, str, dict]) -> None:
        """
        ...
        """
        if isinstance(config, SdgConfig):
            self.config = config
        else:
            self.config = SdgConfig.create_config(config)

        # Create generators
        self._pq_tuple_generator = self.config.pq_tuple_generator.get_generator_class()(
            PQTupleGeneratorArguments(
                sdg_config=self.config
            )
        )
        self.pq_tuples = self._pq_tuple_generator()

        self._pq_function_generator = self.config.pq_function_generator.get_generator_class()(
            PQFunctionGeneratorArguments(
                sdg_config=self.config,
                pq_tuples=self.pq_tuples
            )
        )
        self.pq_functions = self._pq_function_generator()

        # Convert the SDG generated PQ-Tuples and Functions to the new PQ_Relation
        # format to ensure compatibility from SDG- and AIPE-Providers and the
        # representations
        self.pq_relations = PQ_Relation.from_pq_function(
            pq_functions= self.pq_functions,
            pq_tuples= self.pq_tuples,
            config = self.config
        )

        self._knowledge_graph_generator = self.config.knowledge_graph_generator.get_generator_class()(
            KnowledgeGraphGeneratorArguments(
                sdg_config=self.config,
                pq_functions=None,
                pq_tuples=None,
                pq_relations=self.pq_relations
            )
        )

        self._dataset_generator = self.config.dataset_generator.get_generator_class()(
            ExperimentGeneratorArguments(
                sdg_config=self.config,
                pq_functions=self.pq_functions,
                pq_function_generator=self._pq_function_generator,
                pq_tuples=self.pq_tuples
            )
        )

        if self.config.noise_generator is not None:
            self._noise_generator = self.config.noise_generator.get_generator_class()(
                NoiseGeneratorArguments(
                    sdg_config=self.config
                )
            )
        else:
            self._noise_generator = None

        # Create dataset and knowledge graph
        self.create_new_dataset()
        self.create_new_knowledge_graph()

    @property
    def knowledge_graph(self) -> igraph.Graph:
        """
        Knowledge graph that was created using the configured knowledge-graph-generator.
        """
        return deepcopy(self._knowledge_graph)

    @property
    def dataset(self) -> GeneratedDataset:
        """
        Dataset that was created using the configured dataset-generator.
        """
        return self._dataset.copy()

    @property
    def uuid(self) -> str:
        """
        UUID of the dataset. Changes whenever a
        new dataset or knowledge graph is generated.
        """
        return self._id

    @property
    def metadata(self) -> rdflib.Graph:
        """Metadata of the dataset in form of a rdf graph"""
        graph = rdflib.Graph()
        namespace = Namespace("http://purl.org/pdpk/")

        graph.bind("void", VOID)
        graph.bind("dcterms", DCTERMS)
        graph.bind("", namespace)

        dataset = namespace['dataset']
        graph.add((dataset, RDF.type, VOID.Dataset))
        graph.add((dataset, DCTERMS.title, Literal(self.uuid)))
        graph.add((dataset, DCTERMS.source, URIRef('http://purl.org/pdpk')))
        date = Literal(datetime.today().strftime('%Y-%m-%d'), datatype=XSD.date)
        graph.add((dataset, DCTERMS.created, date))
        description = Literal(
            "This is a synthetically generated dataset of parametrisation processes " +
            "in manufacturing. It contains both process data as well as " +
            "knowledge graphs representing the underlying procedural expert knowledge."
        )
        graph.add((dataset, DCTERMS.description, description))
        graph.add((dataset, DCTERMS.license, URIRef("https://opensource.org/licenses/MIT")))
        graph.add((dataset, DCTERMS.subject, URIRef("https://dbpedia.org/page/Manufacturing")))
        triples = Literal(str(len(self.knowledge_graph.es)), datatype=XSD.int)
        graph.add((dataset, VOID.triples, triples))
        entities = Literal(str(len(self.knowledge_graph.vs)), datatype=XSD.int)
        graph.add((dataset, VOID.entities, entities))

        return graph

    def create_new_knowledge_graph(self) -> igraph.Graph:
        """
        Creates a new knowledge graph using the configured knowledge-graph-generator.
        """
        self._knowledge_graph = self._knowledge_graph_generator()
        self._create_new_id()
        return self.knowledge_graph

    def create_new_dataset(self) -> GeneratedDataset:
        """
        Creates a new dataset using the configured dataset-generator.
        """
        self._dataset = self._dataset_generator()
        # Add noise if noise generator is defined
        if self._noise_generator is not None:
            self._noise_generator(self._dataset)
        self._create_new_id()
        return self.dataset

    def _create_new_id(self) -> None:
        """Creates a new uuid for the dataset"""
        self._id = str(uuid.uuid4())
