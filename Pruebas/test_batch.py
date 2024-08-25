# Transformers is tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, and Flax. 
from	transformers				import	CodeAgent			\
									,		Tool
from	IPython.display				import	display

import	os
import	sys

path		= os.path.dirname(__file__)
parent_dir	= os.path.abspath(os.path.join( path , '..'))
sys.path.insert(0, parent_dir)
import	Adapted_data_preprocessing
sys.path.insert(0, path)

class AnomalyBERT_Data_Preprocessing_Tool( Tool ):
	# This class serves to build the preprocessing tool parametrically from the description. This way, it's easy to create variants of this same tool with different descriptions to test.
	# 
	def __init__( self , description ):
		self.name					= "AnomalyBERT_Data_Preprocessing"
		self.description			= description
		self.inputs					= [ "text" ]
		self.outputs				= [ "text" ]

	def __call__( self , dataset , input_dir , output_dir = None , json_dir = None , date_label = None , dataset_mode = None ):

		preprocessed_dataset_folder	= Adapted_data_preprocessing.preprocess_data( dataset , input_dir , output_dir , json_dir , date_label , dataset_mode )

		return preprocessed_dataset_folder

# class AnomalyBERT_Anomaly_Analysis_Tool( Tool ):
# 	# This class serves to build the analysis tool parametrically from the description. This way, it's easy to create variants of this same tool with different descriptions to test.
# 	# 
# 	def __init__( self , description ):
# 		self.name					= "AnomalyBERT_Analyzer"
# 		self.description			= description
# 		self.inputs					= [ "text" ]
# 		self.outputs				= [ "text" ]

# 	# Esto lo hemos añadido para intentar generalizar a diferentes modelos entrenados.
# 	def __init__( self ):
# 		super().__init__(  )

# 		self.is_initialized				= dict()
# 		self.model						= dict()

# 	def setup( self , dataset ):
# 		os.chdir( './AnomalyBERT' )
# 		path							= os.getcwd()
		
# 		if torch.cuda.is_available():
# 			self.device					= torch.device( 'cuda'	)
# 		else:
# 			self.device					= torch.device( 'cpu'	)
 
# 		self.model[ dataset ]			= torch.load( 'logs/best_checkpoints/' + dataset + '_parameters.pt' , map_location = self.device )
# 		os.chdir( os.path.dirname( path ) )
# 		print( "Anomaly BERT model for " + dataset + " loaded: \n")
# 		print( self.model[ dataset ].eval() )
# 		self.is_initialized[ dataset ]	= True

# 	def __call__( self , dataset ):

# 		if not self.is_initialized.get( dataset , False ):
# 			self.setup( dataset )

# 		# Load test dataset.
# 		test_data						= np.load( config.TEST_DATASET[ dataset ] )
# 		test_label						= np.load( config.TEST_LABEL[ dataset ] )

# 		# Data divisions.
# 		test_divisions					= config.DEFAULT_DIVISION[ dataset ]
# 		if test_divisions == 'total':
# 				test_divisions			= [ [ 0 , len( test_data ) ] ]
# 		else:
# 			os.chdir( './AnomalyBERT' )
			
# 			with open( config.DATA_DIVISION[ dataset ][ test_divisions ] , 'r' ) as f:
# 				test_divisions			= json.load( f )
# 			if isinstance( test_divisions , dict ):
# 				test_divisions			= test_divisions.values()

# 			os.chdir( os.path.dirname( path ) )

# 		# Ignore the specific columns.
# 		if dataset in config.IGNORED_COLUMNS.keys():
# 			ignored_column				= np.array( config.IGNORED_COLUMNS[ dataset ] )
# 			remaining_column			= [ col for col in range( len( test_data[ 0 ] ) ) if col not in ignored_column ]
# 			test_data					= test_data[ : , remaining_column ]

# 		# Estimate anomaly scores.
# 		anomaly_scores					= estimate( test_data , self.model[ dataset ] , torch.nn.Sigmoid().to( self.device ) , 1 , 64 , 16 , test_divisions , 5000 , self.device )
# 		anomaly_scores					= anomaly_scores.cpu().numpy()

# 		# Plot data and anomaly scores.
# 		index							= ( 0 , 20000 )	# interval for time steps
# 		data_col_index					= 0	# index of data column

# 		label							= test_label[ index[ 0 ] : index[ 1 ] ].astype( bool )

# 		plt.figure( figsize	= ( 16 , 4 ) )
# 		plt.plot( test_data		[ index[ 0 ] : index[ 1 ] , data_col_index	] , alpha = 0.6 )
# 		plt.scatter	( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
# 					, test_data			[ index[ 0 ] : index[ 1 ] ][ label , data_col_index ]
# 					, c			= 'r'
# 					, s			= 1
# 					, alpha		= 0.8
# 					)
# 		plt.title( 'Original Data' )
# 		plt.show()

# 		plt.figure ( figsize	= ( 16 , 4 ) )
# 		plt.plot( anomaly_scores[ index[ 0 ] : index[ 1 ] , 0				] , alpha = 0.6 )
# 		plt.scatter	( np.arange( index[ 1 ] - index[ 0 ] )[ label ]
# 					, anomaly_scores	[ index[ 0 ] : index[ 1 ] ][ label , 0 ]
# 					, c			= 'r'
# 					, s			= 1
# 					, alpha		= 0.8
# 					)
# 		plt.title( 'Anomaly Scores' )
# 		plt.show()

def tool_variant_agents( variable_tool , tool_description_dict ):
	agent_dict	= {}

	for tool_description_key in tool_description_dict.keys():
		tool_description_indent	= tool_description_dict[ tool_description_key ]
		tool_description		= []

		for line in tool_description_indent:
			tool_description.append( ' '.join( line.split() ) )

		tool_description	= tuple( tool_description )

		agent_dict[ tool_description_key				]	= CodeAgent( tools = [ variable_tool( tool_description			) ]	, add_base_tools = False )
		agent_dict[ tool_description_key + "-INDENT"	]	= CodeAgent( tools = [ variable_tool( tool_description_indent	) ]	, add_base_tools = False )

	return agent_dict

def prompt_tests( agent_dict , agent_variation, prompt_dict ):
	#	This function serves to run all the test propmpts on all the agent instances.
	line_length			= 100
	test_separator		= '-'
	instance_separator	= '='

	for agent_instance_key in agent_dict.keys():
		display( "RESULTS FOR " + agent_variation.upper() + " " + agent_instance_key + ":" )

		for test_prompt_key in prompt_dict.keys():
			display( line_length * test_separator )
			display( "Test prompt " + test_prompt_key + ":" )
			display( agent_dict[ agent_instance_key ].run( prompt_dict[ test_prompt_key ] ) )

		display( line_length * instance_separator )
		display( line_length * instance_separator )
	
	display( "TEST END" )