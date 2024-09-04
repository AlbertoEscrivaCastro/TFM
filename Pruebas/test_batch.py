# Transformers is tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, and Flax. 
from	transformers				import	CodeAgent			\
									,		Tool
from	IPython.display				import	display

import	os
import	sys
import	torch
import	numpy										as	np
import	matplotlib.pyplot							as	plt
import	json

path		= os.path.dirname(__file__)
parent_dir	= os.path.abspath(os.path.join( path , '..'))
sys.path.insert(0, parent_dir)
import	Adapted_utils.Adapted_data_preprocessing	as	adp		# type: ignore
import	Adapted_utils.Adapted_config				as	config	# type: ignore
from	AnomalyBERT.estimate					import	estimate
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

		preprocessed_dataset_folder	= adp.preprocess_data( dataset , input_dir , output_dir , json_dir , date_label , dataset_mode )

		return preprocessed_dataset_folder




class AnomalyBERT_Anomaly_Analysis_Tool( Tool ):
	# This class serves to build the analysis tool parametrically from the description. This way, it's easy to create variants of this same tool with different descriptions to test.
	# 
	def __init__( self , description ):
		self.name					= "AnomalyBERT_Analyzer"
		self.description			= description
		self.inputs					= [ "text" ]
		self.outputs				= [ "text" ]

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
			
			try		: result = agent_dict[ agent_instance_key ].run( prompt_dict[ test_prompt_key ] )
			except	: result = "ERROR: the agent raised a catastrophic error while trying to generate the code to run."

			display( result )

		display( line_length * instance_separator )
		display( line_length * instance_separator )
	
	display( "TEST END" )



# ^(;,;)^ #