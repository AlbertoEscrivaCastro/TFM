# This library adapts the config.py library from AnomalyBERT so it could be used as a library with functions to genetrate the directory routes dinamicaly.
"""Parts of codes are brought from https://github.com/Jhryu30/AnomalyBERT"""

import os

PROJECT_DIR				= os.path.dirname( os.path.abspath( os.path.dirname( __file__ ) ) )
DEFAULT_DATASET_DIR		= '\\Users\\zoiyy\\OneDrive\\Documentos\\DataSets\\TFM\\AnomalyBERT\\processed'
LOG_DIR					= 'logs\\'
DATA_PROPERTY_DIR		= 'data\\'

DATASET_LIST 			= ['SMAP', 'MSL', 'SMD', 'SWaT', 'WADI']

DATA_DIVISION			=	{ 'SMAP'	:	{ 'channel'	: DATA_PROPERTY_DIR + 'SMAP_test_channel.json'
											, 'class'	: DATA_PROPERTY_DIR + 'SMAP_test_class.json'
											}
							, 'MSL'		:	{ 'channel'	: DATA_PROPERTY_DIR + 'MSL_test_channel.json'
											, 'class'	: DATA_PROPERTY_DIR + 'MSL_test_class.json'
											}
							, 'SMD'		:	{ 'channel'	: DATA_PROPERTY_DIR + 'SMD_test_channel.json'
											}
							}

DEFAULT_DIVISION		=	{ 'SMAP'	: 'channel'
							, 'MSL'		: 'channel'
							, 'SMD'		: 'channel'
							, 'SWaT'	: 'total'
							, 'WADI'	: 'total'
							}

NUMERICAL_COLUMNS		=	{ 'SMAP'	: ( 0 , )
							, 'MSL'		: ( 0 , )
							, 'SMD' 	: tuple( list( range( 7 ) ) + list( range( 8 , 38 ) ) )
							}

CATEGORICAL_COLUMNS		=	{ 'SMAP'	: range( 1 , 25 )
							, 'MSL'		: range( 1, 55 )
							, 'SMD'		: ( 7 , )
							, 'SWaT'	: tuple( [ 2 , 3 , 4 , 9 ] + list( range( 11 , 16 ) ) + list( range( 19 , 25 ) )	\
										+ list( range( 29 , 34 ) ) + [ 42 , 43 , 48 , 49 , 50 ] )
							, 'WADI'	: tuple( [ 6 , 7 ] + list( range( 9 , 19 ) ) + list( range( 47 , 59 ) )				\
										+ list( range( 68 , 81 ) ) + [ 82 , 84 , 87 ] + list( range( 91 , 97 ) )			\
										+ [ 111 ] + list( range( 113 , 120 ) ) + [ 121 ] )
							}

IGNORED_COLUMNS			=	{ 'SWaT'	: ( 10 , )
							, 'WADI'	: ( 102 , )
							}

def set_directory( custom_dataset_dir = None ) :
	global DATASET_DIR , TRAIN_DATASET , TEST_DATASET , TEST_LABEL , DEFAULT_DIVISION , NUMERICAL_COLUMNS , CATEGORICAL_COLUMNS

	if custom_dataset_dir == None :
		DATASET_DIR		= DEFAULT_DATASET_DIR

	else :
		DATASET_DIR		= custom_dataset_dir
        
	TRAIN_DATASET		= {}
	TEST_DATASET		= {}
	TEST_LABEL			= {}

	for data_name in DATASET_LIST:
		TRAIN_DATASET[ data_name ]	= os.path.join( DATASET_DIR , data_name + '_train.npy'		)
		TEST_DATASET[ data_name ]	= os.path.join( DATASET_DIR , data_name + '_test.npy'		)
		TEST_LABEL[ data_name ]		= os.path.join( DATASET_DIR , data_name + '_test_label.npy'	)

	# SMD series
	train_smd			= { 'SMD{}'.format( i ) : os.path.join( DATASET_DIR , 'SMD{}_train.npy'.format( i )			)	for i in range( 28 ) }
	test_smd			= { 'SMD{}'.format( i ) : os.path.join( DATASET_DIR , 'SMD{}_test.npy'.format( i )			)	for i in range( 28 ) }
	label_smd			= { 'SMD{}'.format( i ) : os.path.join( DATASET_DIR , 'SMD{}_test_label.npy'.format( i )	)	for i in range( 28 ) }
	division_smd		= { 'SMD{}'.format( i ) : 'total'																for i in range( 28 ) }
	numerical_smd		= { 'SMD{}'.format( i ) : NUMERICAL_COLUMNS[ 'SMD' ]											for i in range( 28 ) }
	categorical_smd		= { 'SMD{}'.format( i ) : ( 7 , )																for i in range( 28 ) }

	TRAIN_DATASET.update( train_smd )
	TEST_DATASET.update( test_smd )
	TEST_LABEL.update( label_smd )
	DEFAULT_DIVISION.update( division_smd )
	NUMERICAL_COLUMNS.update( numerical_smd )
	CATEGORICAL_COLUMNS.update( categorical_smd )

	# SWaT and WADI update
	NUMERICAL_COLUMNS[ 'SWaT' ]	= tuple( [ i for i in range( 0 , 51 )	if	( i not in CATEGORICAL_COLUMNS[ 'SWaT' ] ) \
																		and ( i not in IGNORED_COLUMNS[ 'SWaT' ] ) ] )
	NUMERICAL_COLUMNS[ 'WADI' ]	= tuple( [ i for i in range( 0 , 123 )	if	( i not in CATEGORICAL_COLUMNS[ 'WADI' ] ) \
																		and ( i not in IGNORED_COLUMNS[ 'WADI' ] ) ] )



# ^(;,;)^ #