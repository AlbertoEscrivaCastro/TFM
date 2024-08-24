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

# def tool_variant_agents( variable_tool , tool_description_list ):
# 	agent_list	= []

# 	for tool_description in tool_description_list:
# 		agent_list.append( CodeAgent( tools = [ variable_tool( tool_description ) ] , add_base_tools = False ) )

# 	return agent_list

# def prompt_tests( agent_list , agent_variation, prompt_list ):
# 	#	This function serves to run all the test propmpts on all the agent instances.
# 	line_length			= 100
# 	test_separator		= '-'
# 	instance_separator	= '='

# 	for i , agent_instance in enumerate( agent_list ):
# 		display( "RESULTS FOR " + agent_variation.upper() + " " + str( i + 1 ) + ":" )

# 		for j , test_prompt in enumerate( prompt_list ):
# 			display( line_length * test_separator )
# 			display( "Test prompt " + str( j + 1 ) + ":" )
# 			display( agent_instance.run( test_prompt ) )

# 		display( line_length * instance_separator )
# 		display( line_length * instance_separator )
	
# 	display( "TEST END" )