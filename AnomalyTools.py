"""Parts of codes are brought from https://github.com/Jhryu30/AnomalyBERT"""
import	requests
from	PIL											import	Image
from	transformers								import	Tool

import	os
import	sys
import	inspect
import	json
import	torch
import	numpy										as		np
import	matplotlib.pyplot							as		plt
# %matplotlib inline

# import	Adapted_data_preprocessing
import	Adapted_utils.Adapted_data_preprocessing	as		adp # type: ignore
import	Adapted_utils.Adapted_config				as		config

os.chdir( './AnomalyBERT' )
path									= os.getcwd()
#	Habrá que cambiarlo por el adapted, pero creía que ya estaba hecho. ¿Dónde lo estamos usando?ç
#	Lo estamos usando en Experimentos2 y Experimentos3.
# import	AnomalyBERT.utils.config	as		config
from	AnomalyBERT.estimate		import	estimate
from	AnomalyBERT.compute_metrics	import	f1_score
os.chdir( os.path.dirname( path ) )


def toolbox():
	# return [ tool[ 1 ]() for tool in inspect.getmembers( sys.modules[__name__] ) if tool[ 1 ].__name__ == "AnomalyTools" ]
	return [ tool[ 1 ]() for tool in inspect.getmembers( sys.modules[ __name__ ] , inspect.isclass )  if tool[ 1 ].__module__ == "AnomalyTools" ]


class CatImageFetcher( Tool ):
	name								= "cat_fetcher"
	description							= ("This is a tool that fetches an actual image of a cat online. It takes no input, and returns the image of a cat.")

	inputs								= []
	outputs								= [ "text" ]

	def __call__( self ):
		return Image.open( requests.get( 'https://cataas.com/cat' , stream = True ).raw ).resize( ( 256 , 256 ) )



class AnomalyBERT_Analyzer( Tool ):
	# Habrá que adaptarlo para que pueda utilizar directorios que no se encuentren en el config. Para experimentar, se ha hecho la copia AnomalyBERT_Analyzer_BIS(Tool)
	name								= "AnomalyBERT_Analyzer"
	description							= ( "This is a tool that identifies anomalies in temporal data. It takes as input the type of dataset (implemented for SWaT/SMAP/MSL/WADI) and returns its plot with the anomalies higlighted." )

	inputs								= [ "text" ]
	outputs								= [ "text" ]

	# Esto lo hemos añadido para intentar generalizar a diferentes modelos entrenados.
	def __init__( self ):
		super().__init__(  )

		self.is_initialized				= dict()
		self.model						= dict()

	def setup( self , dataset ):
		os.chdir( './AnomalyBERT' )
		path							= os.getcwd()
		
		if torch.cuda.is_available():
			self.device					= torch.device( 'cuda'	)
		else:
			self.device					= torch.device( 'cpu'	)
 
		self.model[ dataset ]			= torch.load( 'logs/best_checkpoints/' + dataset + '_parameters.pt' , map_location = self.device )
		os.chdir( os.path.dirname( path ) )
		print( "Anomaly BERT model for " + dataset + " loaded: \n")
		print( self.model[ dataset ].eval() )
		self.is_initialized[ dataset ]	= True

# ¿referencia https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/tools/text_summarization.py ?
	# https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/agent#transformers.Tool
	def __call__( self , dataset ):

		if not self.is_initialized.get( dataset , False ):
			self.setup( dataset )

		# Load test dataset.
		test_data						= np.load( config.TEST_DATASET[ dataset ] )
		test_label						= np.load( config.TEST_LABEL[ dataset ] )

		# Data divisions.
		test_divisions					= config.DEFAULT_DIVISION[ dataset ]
		if test_divisions == 'total':
				test_divisions			= [ [ 0 , len( test_data ) ] ]
		else:
			os.chdir( './AnomalyBERT' )
			
			with open( config.DATA_DIVISION[ dataset ][ test_divisions ] , 'r' ) as f:
				test_divisions			= json.load( f )
			if isinstance( test_divisions , dict ):
				test_divisions			= test_divisions.values()

			os.chdir( os.path.dirname( path ) )

		# Ignore the specific columns.
		if dataset in config.IGNORED_COLUMNS.keys():
			ignored_column				= np.array( config.IGNORED_COLUMNS[ dataset ] )
			remaining_column			= [ col for col in range( len( test_data[ 0 ] ) ) if col not in ignored_column ]
			test_data					= test_data[ : , remaining_column ]

		# Estimate anomaly scores.
		anomaly_scores					= estimate( test_data , self.model[ dataset ] , torch.nn.Sigmoid().to( self.device ) , 1 , 64 , 16 , test_divisions , 5000 , self.device )
		anomaly_scores					= anomaly_scores.cpu().numpy()

		# Plot data and anomaly scores.
		index							= ( 0 , 20000 )	# interval for time steps
		data_col_index					= 0	# index of data column

		label							= test_label[ index[ 0 ] : index[ 1 ] ].astype( bool )

		plt.figure( figsize	= ( 16 , 4 ) )
		plt.plot( test_data		[ index[ 0 ] : index[ 1 ] , data_col_index	] , alpha = 0.6 )
		plt.scatter	( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
					, test_data			[ index[ 0 ] : index[ 1 ] ][ label , data_col_index ]
					, c			= 'r'
					, s			= 1
					, alpha		= 0.8
					)
		plt.title( 'Original Data' )
		plt.show()

		plt.figure ( figsize	= ( 16 , 4 ) )
		plt.plot( anomaly_scores[ index[ 0 ] : index[ 1 ] , 0				] , alpha = 0.6 )
		plt.scatter	( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
					, anomaly_scores	[ index[ 0 ] : index[ 1 ] ][ label , 0 ]
					, c			= 'r'
					, s			= 1
					, alpha		= 0.8
					)
		plt.title( 'Anomaly Scores' )
		plt.show()

# class AnomalyBERTAnalyzer(PipelineTool):
#	 default_checkpoint = "google/flan-t5-base"
#	 description = (
#	   	his is a tool that answers questions related to a text. It takes two arguments named `text`, which is the "
#	   	ext where to find the answer, and `question`, which is the question, and returns the answer to the question."
#	 )
#	 name = "text_qa"
#	 pre_processor_class = AutoTokenizer
#	 model_class = AutoModelForSeq2SeqLM

#	 inputs = ["text", "text"]
#	 outputs = ["text"]

#	 def encode(self, text: str, question: str):
#	   	ompt = QA_PROMPT.format(text=text, question=question)
#	   	turn self.pre_processor(prompt, return_tensors="pt")

#	 def forward(self, inputs):
#	   	tput_ids = self.model.generate(**inputs)

#	   	_b, _ = inputs["input_ids"].shape
#	   	t_b = output_ids.shape[0]

#	   	turn output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])[0][0]

#	 def decode(self, outputs):
#	   	turn self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
		
class AnomalyBERT_Analyzer_BIS( Tool ):
	# Habrá que adaptarlo para que pueda utilizar directorios que no se encuentren en el config. Es una copia de AnomalyBERT_Analyzer(Tool) para experimentar.
	# ¿Quizá usar la herramienta tal como estaba y crear un config alternativo?
	# Algo de adaptación seguirá siendo necesaria de todos modos...
	name								= "AnomalyBERT_Analyzer_BIS"
	description							= ( "This is a tool that identifies anomalies in temporal data. It takes as input the type of dataset (implemented for SWaT/SMAP/MSL/WADI) and optionaly the route of the preprocessed dataset folder, and returns its plot with the anomalies higlighted." )

	inputs								= [ "text" ]
	outputs								= [ "text" ]

	# Esto lo hemos añadido para intentar generalizar a diferentes modelos entrenados.
	def __init__( self ):
		super().__init__(  )

		self.is_initialized				= dict()
		self.model						= dict()

	def setup( self , dataset_type ):
		os.chdir( './AnomalyBERT' )
		path							= os.getcwd()
		
		if torch.cuda.is_available():
			self.device					= torch.device( 'cuda'	)
		else:
			self.device					= torch.device( 'cpu'	)
 
		self.model[ dataset_type ]			= torch.load( 'logs/best_checkpoints/' + dataset_type + '_parameters.pt' , map_location = self.device )
		os.chdir( os.path.dirname( path ) )
		print( "Anomaly BERT model for " + dataset_type + " loaded: \n")
		print( self.model[ dataset_type ].eval() )
		self.is_initialized[ dataset_type ]	= True

# ¿referencia https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/tools/text_summarization.py ?
	# https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/agent#transformers.Tool
	def __call__( self , dataset_type , preprocessed_dataset_folder = None ):

		if not self.is_initialized.get( dataset_type , False ):
			self.setup( dataset_type )

		# Load test dataset.
		if preprocessed_dataset_folder is None:
			test_data					= np.load( config.TEST_DATASET[ dataset_type ] )
			test_label					= np.load( config.TEST_LABEL[ dataset_type ] )
		else:
			test_data					= np.load( os.path.join( preprocessed_dataset_folder , dataset_type + '_test.npy'		) )
			test_label					= np.load( os.path.join( preprocessed_dataset_folder , dataset_type + '_test_label.npy'	) )

		# Data divisions.
		test_divisions					= config.DEFAULT_DIVISION[ dataset_type ]
		if test_divisions == 'total':
				test_divisions			= [ [ 0 , len( test_data ) ] ]
		else:
			os.chdir( './AnomalyBERT' )
			
			with open( config.DATA_DIVISION[ dataset_type ][ test_divisions ] , 'r' ) as f:
				test_divisions			= json.load( f )
			if isinstance( test_divisions , dict ):
				test_divisions			= test_divisions.values()

			os.chdir( os.path.dirname( path ) )

		# Ignore the specific columns.
		if dataset_type in config.IGNORED_COLUMNS.keys():
			ignored_column				= np.array( config.IGNORED_COLUMNS[ dataset_type ] )
			remaining_column			= [ col for col in range( len( test_data[ 0 ] ) ) if col not in ignored_column ]
			test_data					= test_data[ : , remaining_column ]

		# Estimate anomaly scores.
		anomaly_scores					= estimate( test_data , self.model[ dataset_type ] , torch.nn.Sigmoid().to( self.device ) , 1 , 64 , 16 , test_divisions , 5000 , self.device )
		anomaly_scores					= anomaly_scores.cpu().numpy()

		# Plot data and anomaly scores.
		index							= ( 0 , 20000 )	# interval for time steps
		data_col_index					= 0	# index of data column

		label							= test_label[ index[ 0 ] : index[ 1 ] ].astype( bool )

		plt.figure( figsize	= ( 16 , 4 ) )
		plt.plot( test_data		[ index[ 0 ] : index[ 1 ] , data_col_index	] , alpha = 0.6 )
		plt.scatter	( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
					, test_data			[ index[ 0 ] : index[ 1 ] ][ label , data_col_index ]
					, c			= 'r'
					, s			= 1
					, alpha		= 0.8
					)
		plt.title( 'Original Data' )
		plt.show()

		plt.figure ( figsize	= ( 16 , 4 ) )
		plt.plot( anomaly_scores[ index[ 0 ] : index[ 1 ] , 0				] , alpha = 0.6 )
		plt.scatter	( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
					, anomaly_scores	[ index[ 0 ] : index[ 1 ] ][ label , 0 ]
					, c			= 'r'
					, s			= 1
					, alpha		= 0.8
					)
		plt.title( 'Anomaly Scores' )
		plt.show()


class AnomalyBERT_Data_Preprocessing( Tool ):
	# This tool serves as an interface between the Agent and the Adapted_data_preprocessing.py library.
	# 
	name								= "AnomalyBERT_Data_Preprocessing"
	description							= \
("This auxiliary tool is designed to support the primary process of satellite telemetry data analysis for anomaly detection using AnomalyBERT.\
 The specific task of this tool is to preprocess the dataset, preparing it for subsequent analysis.\
 This involves cleaning, normalizing, and transforming the raw telemetry data to ensure it is in an optimal format for the AnomalyBERT detection process."
,"Mandatory Input Parameters:\
 'dataset' (the type of dataset, permitted options are SWaT, SMAP, MSL, and WADI),\
 'input_dir' (the input directory of the raw data, it must be a path written as a raw string).\
 Optional Input Parameters (when using this function, optional parameters should be assigned values using keyword arguments;\
 only provide these keyword arguments if their values are needed; otherwise, omit them entirely):\
 'output_dir' (the output directory for the preprocessed data, it must be a path written as a raw string),\
 'json_dir' (the directory where the JSON files will be saved if created. It must be a path written as a raw string),\
 'date_label' (the date label for WADI datasets)\
 'dataset_mode' (The 'dataset_mode' parameter specifies the intended use and purpose of the dataset and can take one of the following values:\
 'train', indicates that the dataset will be used for training the neural network, so it is labeled and provides the train and test partitions;\
 'test', indicates that the dataset will be used just for testing the already trained neural network, so it is also labeled but only has the test partion;\
 'exploitation', indicates that the dataset is intended to be analyzed and detect its anomalies (inference), so it's not labeled.)."
,"As output, it returns the path of the files where the preprocessed data has been stored."
)

	inputs								= [ "text" ]
	outputs								= [ "text" ]

# ¿referencia https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/tools/text_summarization.py ?
	# https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/agent#transformers.Tool
	def __call__( self , dataset , input_dir , output_dir = None , json_dir = None , date_label = None , dataset_mode = None ):

		preprocessed_dataset_folder = adp.preprocess_data( dataset , input_dir , output_dir , json_dir , date_label , dataset_mode )

		return preprocessed_dataset_folder