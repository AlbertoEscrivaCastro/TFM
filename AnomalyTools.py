"""Parts of codes are brought from https://github.com/Jhryu30/AnomalyBERT"""
import	requests														
from	PIL										import	Image
from	transformers							import	Tool

import	os
import	sys
import	inspect
import	json
import	torch
import	numpy										as	np
import	matplotlib.pyplot							as	plt
# %matplotlib inline

# import	Adapted_data_preprocessing
import	Adapted_utils.Adapted_data_preprocessing	as	adp			# type: ignore
import	Adapted_utils.Adapted_config				as	config

os.chdir( './AnomalyBERT' )
#	Habrá que cambiarlo por el adapted, pero creía que ya estaba hecho. ¿Dónde lo estamos usando?
#	Lo estamos usando en Experimentos2 y Experimentos3.
# import	AnomalyBERT.utils.config	as		config
from	AnomalyBERT.estimate					import	estimate
from	AnomalyBERT.compute_metrics				import	f1_score
os.chdir( os.path.dirname( os.getcwd() ) )



def toolbox():
	# return [ tool[ 1 ]() for tool in inspect.getmembers( sys.modules[__name__] ) if tool[ 1 ].__name__ == "AnomalyTools" ]
	return [ tool[ 1 ]() for tool in inspect.getmembers( sys.modules[ __name__ ] , inspect.isclass )  if tool[ 1 ].__module__ == "AnomalyTools" ]



class CatImageFetcher( Tool ):
	# This tools is not used for the current aplication. 
	# The only purpose to include it is for it to serve as a check while developing.
	# If the agent invokes it correctly, it means the whole toolbox is available, so problems providing the list of tools can be discarded as the cause for any issue that might arise.
	name								= "cat_fetcher"
	description							= ("This is a tool that fetches an actual image of a cat online. It takes no input, and returns the image of a cat.")

	inputs								= []
	outputs								= [ "text" ]

	def __call__( self ):
		return Image.open( requests.get( 'https://cataas.com/cat' , stream = True ).raw ).resize( ( 256 , 256 ) )



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



class AnomalyBERT_Analyzer( Tool ):
	name								= "AnomalyBERT_Analyzer"
	description							= \
("This tool is the primary process of satellite telemetry data analysis for anomaly detection using AnomalyBERT,\
 which needs the data to be already prerocessed.\
 It takes as input the type of dataset (implemented for SWaT/SMAP/MSL/WADI) and returns its plot with the anomalies higlighted."
,"Mandatory Input Parameters:\
 - 'dataset_type' (the type of dataset, permitted options are SWaT, SMAP, MSL, and WADI),\
 Optional Input Parameters (when using this function, optional parameters should be assigned values using keyword arguments;\
 only provide these optional parameters if their values are needed; otherwise, omit them entirely):\
 - 'preprocessed_dataset_folder' (the input directory of the preprocessed data, it must be a path written as a raw string).\
 - 'dataset_mode' (The 'dataset_mode' parameter specifies the intended use and purpose of the dataset and can take one of the following values:\
 * 'train', indicates that the dataset will be used for training the neural network, so it is labeled and provides the train and test partitions;\
 * 'test', indicates that the dataset will be used just for testing the already trained neural network, so it is also labeled but only has the test partion;\
 * 'exploitation', indicates that the dataset is intended to be analyzed and detect its anomalies (inference), so it's not labeled.)."
,"As output, it returns the path of the files where the anomaly scores have been stored after the analysis and plots the results."
)

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
	def __call__( self , dataset_type , preprocessed_dataset_folder = None , analysis_output_file = None  , dataset_mode = None ):

		if not self.is_initialized.get( dataset_type , False ):
			self.setup( dataset_type )

		# Set directories.
		config.set_directory( preprocessed_dataset_folder )

		# Load test dataset.
		test_data						= np.load( config.TEST_DATASET[ dataset_type ] )
		
		if dataset_mode == 'train'	\
		or dataset_mode == 'test'	:
			test_label					= np.load( config.TEST_LABEL[ dataset_type ] )

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

			os.chdir( os.path.dirname( os.getcwd() ) )

		# Ignore the specific columns.
		if dataset_type in config.IGNORED_COLUMNS.keys():
			ignored_column				= np.array( config.IGNORED_COLUMNS[ dataset_type ] )
			remaining_column			= [ col for col in range( len( test_data[ 0 ] ) ) if col not in ignored_column ]
			test_data					= test_data[ : , remaining_column ]

		# Estimate anomaly scores.
		anomaly_scores					= estimate( test_data , self.model[ dataset_type ] , torch.nn.Sigmoid().to( self.device ) , 1 , 64 , 16 , test_divisions , 5000 , self.device )
		anomaly_scores					= anomaly_scores.cpu().numpy()

		# Store the estimations.
		# Esto... esto no está bien... falta averiguar qué meter en lo del state_dict y quizá ya esté, pero me preocupa lo del model.eval() del main de estimate...
		analysis_output_file			= config.TEST_DATASET[ dataset_type ][ : -4 ] + '_results.npy' if analysis_output_file == None else analysis_output_file
		print( analysis_output_file )
		np.save( analysis_output_file , anomaly_scores )
		
		# Plot data and anomaly scores.
		index							= ( 0 , 20000 )	# interval for time steps
		data_col_index					= 0	# index of data column

		if dataset_mode == 'train'	\
		or dataset_mode == 'test'	:
			label						= test_label[ index[ 0 ] : index[ 1 ] ].astype( bool )

		# Adapted plots so both plots are displayed on the same figure. Still need to test if it works.
		plt.figure( figsize	= ( 16 , 4 ) )
		fig , ax1						= plt.subplots()

		color							= 'tab:green'
		ax1.set_xlabel( 'time' )
		ax1.set_ylabel( 'Original Data'		, color = color )
		ax1.plot( test_data		[ index[ 0 ] : index[ 1 ] , data_col_index	] , color = color , alpha = 0.6 )
		if dataset_mode == 'train'	\
		or dataset_mode == 'test'	:
			plt.scatter ( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
						, test_data			[ index[ 0 ] : index[ 1 ] ][ label , data_col_index ]
						, c			= 'r'
						, s			= 1
						, alpha		= 0.8
						)
		ax1.tick_params( axis = 'y' , labelcolor = color )

		ax2 = ax1.twinx()  # Instantiate a second Axes that shares the same x-axis

		color = 'tab:blue'
		ax2.set_ylabel( 'Anomaly Scores'	, color = color )
		ax2.plot( anomaly_scores[ index[ 0 ] : index[ 1 ] , 0				]  , color = color , alpha = 0.6 )
		if dataset_mode == 'train'	\
		or dataset_mode == 'test'	:
			plt.scatter	( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
						, anomaly_scores	[ index[ 0 ] : index[ 1 ] ][ label , 0 ]
						, c			= 'r'
						, s			= 1
						, alpha		= 0.8
						)
		ax2.tick_params( axis = 'y' , labelcolor = color )

		fig.tight_layout()
		plt.show()

		return analysis_output_file

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



# ^(;,;)^ #