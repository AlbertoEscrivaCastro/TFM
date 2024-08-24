# This library/script adapts the data_preprocessing.py script from AnomalyBERT so it could be used as a library by the appropiate tool to genetrate the directory routes dinamicaly.
# It also takes the date lable as imput for the WADI datasets, so it is not hardcoded here.
"""Parts of codes are brought from https://github.com/NetManAIOps/OmniAnomaly"""
"""Parts of codes are brought from https://github.com/Jhryu30/AnomalyBERT"""

import	ast
import	csv
import	os
# import	sys
from	pickle					import dump
import	json

import	argparse

import	numpy	as np
import	pandas	as pd
from	sklearn.preprocessing	import MinMaxScaler

datasets	=	[ 'SMD'
				, 'SMAP'
				, 'MSL'
				, 'SWaT'
				, 'WADI'
				]


# Hay que comprobar si realmente dataset y output_folder se necesitan aquí y en caso de hacerlo, informarlas apropiadamente.
def load_as_np( category , filename , dataset , dataset_folder,  output_folder ):
	temp = np.genfromtxt( os.path.join( dataset_folder , category , filename ) ,
						 dtype = np.float32 ,
						 delimiter = ',' )
	return temp



def load_data( dataset , base_dir , output_folder , json_folder , date_label , process_type ) :

	if process_type == 'train'	:
		output_train_file_name		= os.path.join( output_folder	, dataset + "_train.npy"			)
	else :
		output_train_file_name		= None

	if process_type == 'train'	\
	or process_type == 'test'	:
		output_test_label_file_name	= os.path.join( output_folder	, dataset + "_test_label.npy"		)

	else :
		output_test_label_file_name	= None

	output_test_file_name			= os.path.join( output_folder	, dataset + "_test.npy"				)
	json_test_channel_file_name		= os.path.join( json_folder		, dataset + "_test_channel.json"	)

	if dataset == 'SMD':
		dataset_folder				= os.path.join( base_dir , 'OmniAnomaly/ServerMachineDataset' )
		file_list					= os.listdir( os.path.join( dataset_folder, "test"	) )
		
		train_files					= []
		test_files					= []
		label_files					= []
		file_length					= [ 0 ]

		for filename in file_list:
			if filename.endswith( '.txt' )	:
				if process_type == 'train'	:
					train_files.append( load_as_np( 'train'			, filename , filename.strip( '.txt' ) , dataset_folder , output_folder ) )
				
				if process_type == 'train'	\
				or process_type == 'test'	:
					label_files.append( load_as_np( 'test_label'	, filename , filename.strip( '.txt' ) , dataset_folder , output_folder ) )
				
				test_files.append( load_as_np( 'test' , filename , filename.strip( '.txt' ) , dataset_folder , output_folder ) )
				file_length.append( len( test_files[ -1 ] ) )
		
		for i, test in zip( range( len( test_files ) ) , test_files ) :
			if process_type == 'train'	:
				np.save( os.path.join( output_folder , dataset + "{}_train.npy".format( i )			) , train_files[ i ]	)
			
			if process_type == 'train'	\
			or process_type == 'test'	:
				np.save( os.path.join( output_folder , dataset + "{}_test_label.npy".format( i )	) , label_files[ i ]	)
			
			np.save( os.path.join( output_folder , dataset + "{}_test.npy".format( i )			) , test	)
			

		if process_type == 'train'	:	
			train_files	= np.concatenate( train_files	, axis = 0 )
			np.save( output_train_file_name , train_files	)
		
		if process_type == 'train'	\
		or process_type == 'test'	:
			label_files	= np.concatenate( label_files	, axis = 0 )
			np.save( output_test_label_file_name , label_files	)

		test_files			= np.concatenate( test_files	, axis = 0 )
		np.save( output_test_file_name , test_files	)

		file_length			= np.cumsum( np.array( file_length ) ).tolist()
		channel_divisions	= []

		for i in range( len( file_length ) - 1 ) :
			channel_divisions.append( [ file_length[ i ] , file_length[ i + 1 ] ] )
		
		with open( json_test_channel_file_name , 'w' ) as file :
			json.dump( channel_divisions , file )


	elif dataset == 'SMAP'	\
	  or dataset == 'MSL'	:
		dataset_folder		= os.path.join( base_dir , 'telemanom/data' )

		labels				= []
		class_divisions		= {}
		channel_divisions	= []
		current_index		= 0

		if process_type == 'train'	\
		or process_type == 'test'	:
			with open( os.path.join( dataset_folder , 'labeled_anomalies.csv' ) , 'r' ) as file :
				csv_reader	= csv.reader( file, delimiter = ',' )
				res			= [ row for row in csv_reader ][ 1 : ]
			
			res				= sorted( res , key = lambda k : k[ 0 ][ 0 ] + '-{:2d}'.format( int( k[ 0 ][ 2 : ] ) ) )

			data_info		= [ row for row in res if row[ 1 ] == dataset and row[ 0 ] != 'P-2' ]

		#	¡¡¡ OJO !!! Relacionado con el resto de OJOs
		#	Hay que ver si esto se usa sólo para el test o si se usa también para el procesamiento de producción.
		#	Si también PRODUCCIÓN (exploitation), sacar del IF TRAIN OR TEST y usar el método de la tipología anterior, leyendo los nombres de ficheros de TEST. Ahora emplea la primera columna del CSV para tener una lista de los canales.
		#	Este FOR y los dos siguientes WITH OPEN, habría que separar del for cosas que se tendrían qeu quedar aquí (labels) y cosas que tendrían que salir fuera del IF TRAIN OR TEST (channels ¿y class divisions?).
		#	Este es el último OJO que queda, el resto se ha decidido trasladar los elementos fuera del if como si se fuesen a usar durante "producción" porque se puede obtener los mismos datos sin usar los datos de etiquetas.
		#	Este se ha dejado así por la complejidad que implicaría separar las tareas específicas para labels de las que se pueden obtener de los datos de test.
			for row in data_info:
				anomalies	= ast.literal_eval( row[ 2 ] )
				length		= int( row[ -1 ] )
				label		= np.zeros( [ length ] , dtype = bool )

				for anomaly in anomalies:
					label[ anomaly[ 0 ] : anomaly[ 1 ] + 1 ] = True
				
				labels.extend( label )
				
				_class = row[ 0 ][ 0 ]
				if _class in class_divisions.keys() :
					class_divisions[ _class ][ 1 ] += length
				else:
					class_divisions[ _class ] = [ current_index , current_index + length ]
				
				channel_divisions.append( [ current_index , current_index + length ] )
				current_index += length
			
			with open( os.path.join( json_folder , dataset + "_" + 'test_class.json'	) , 'w' ) as file :
				json.dump( class_divisions		, file )
			
			with open( json_test_channel_file_name , 'w' ) as file :
				json.dump( channel_divisions	, file )

			labels = np.asarray( labels )
			np.save( output_test_label_file_name	, labels				)

		def concatenate( category ) :
			data = []
			for row in data_info:
				filename	= row[ 0 ]
				temp		= np.load( os.path.join( dataset_folder , category , filename + '.npy' ) )
				data.extend( temp )
			
			data = np.asarray( data )
			data = MinMaxScaler().fit_transform( data )
		
		if process_type == 'train'	:
			np.save( output_train_file_name		, concatenate( 'train'	)	)
			
		np.save( output_test_file_name		, concatenate( 'test'	)	)
	
	elif dataset == 'SWaT' :
		dataset_folder	= os.path.join( base_dir , 'SWaT/Physical' )

		if process_type == 'train'	:
			normal_data		= pd.read_excel( os.path.join( dataset_folder , 'SWaT_Dataset_Normal_v1.xlsx' ) )
			normal_data		= normal_data.iloc[ 1 : , 1 : -1 ].to_numpy()
			normal_data		= MinMaxScaler().fit_transform( normal_data		).clip( 0 , 1 )
			np.save( output_train_file_name , normal_data		)
		
		abnormal_data	= pd.read_excel( os.path.join( dataset_folder , 'SWaT_Dataset_Attack_v0.xlsx' ) )

		if process_type == 'train'	\
		or process_type == 'test'	:
			abnormal_label	= abnormal_data.iloc[ 1 : , -1		] == 'Attack'
			abnormal_label	= abnormal_label.to_numpy().astype( int )
			np.save( output_test_label_file_name	, abnormal_label	)
		
		abnormal_data	= abnormal_data.iloc[ 1 : , 1 : -1	].to_numpy()
		abnormal_data	= MinMaxScaler().fit_transform( abnormal_data	).clip( 0 , 1 )
		np.save( output_test_file_name			, abnormal_data		)
		
		
	elif dataset == 'WADI':
		if process_type == 'train'	:
			normal_data		= pd.read_csv( os.path.join( base_dir , 'WADI/WADI.' + date_label + '/WADI_14days_new.csv'		) )
			normal_data		= normal_data.dropna( axis = 'columns' , how = 'all' ).dropna()
			normal_data		= normal_data.iloc[ : , 3 : ].to_numpy()
			normal_data		= MinMaxScaler().fit_transform( normal_data ).clip( 0 , 1 )
			np.save( output_train_file_name , normal_data		)
		
		abnormal_data	= pd.read_csv( os.path.join( base_dir , 'WADI/WADI.' + date_label + '/WADI_attackdataLABLE.csv'	) , header = 1	)
		abnormal_data	= abnormal_data.dropna( axis = 'columns' , how = 'all' ).dropna()

		if process_type == 'train'	\
		or process_type == 'test'	:
			abnormal_label	= abnormal_data.iloc[ : , -1		] == -1
			abnormal_label	= abnormal_label.to_numpy().astype( int )
			np.save( output_test_label_file_name	, abnormal_label	)
		
		abnormal_data	= abnormal_data.iloc[ : , 3 : -1	].to_numpy()
		abnormal_data	= MinMaxScaler().fit_transform( abnormal_data ).clip( 0 , 1 )
		np.save( output_test_file_name			, abnormal_data		)



def preprocess_data( dataset , data_dir , out_dir = None , json_dir = None , date_label = None , process_type = None ):
	if dataset in datasets:
		if out_dir == None :
			output_folder	= os.path.join( data_dir , 'processed' )
		else:
			output_folder	= out_dir
		
		if not os.path.exists( output_folder ) :
			os.mkdir( output_folder )
		
		if json_dir == None :
			json_folder		= os.path.join( data_dir , 'json' )
		else:
			json_folder		= json_dir
		
		if not os.path.exists( json_folder ):
			os.mkdir( json_folder )

		if	dataset		== 'WADI'	\
		and	date_label	== None :
			date_label		= 'A2_19 Nov 2019'

		if process_type	== None :
			process_type = "exploitation"

		# It could be interesting to validate date_label content.
		load_data( dataset , data_dir , output_folder , json_folder , date_label , process_type )

		return output_folder



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument( "--dataset"		, required	= True	, type = str ,
						help = "Name of dataset; SMD/SMAP/MSL/SWaT/WADI"						)
	parser.add_argument( "--data_dir"		, required	= True	, type = str ,
						help = "Directory of raw data"											)
	parser.add_argument( "--out_dir"		, default	= None	, type = str ,
						help = "Directory of the processed data"								)
	parser.add_argument( "--json_dir"		, default	= None	, type = str ,
						help = "Directory of the json files for the processed data"				)
	parser.add_argument( "--date_label"		, default	= None	, type = str ,
						help = "Date label for WADI files in the shape 'A2_19 Nov 2019'"		)
	parser.add_argument( "--process_type"	, default	= None	, type = str ,
						help = "Type of the preproccessing to perform; train/test/exploitation"	)
	options = parser.parse_args()

	preprocess_data( options.dataset , options.data_dir , options.out_dir , options.json_dir , options.date_label , options.process_type )
