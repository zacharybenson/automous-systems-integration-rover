from tf.keras.utils import sequences



class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
    	DEFAULT_DATA_PATH, 
		offsets,
		sequence_size,
		shuffle_series,
		random_State,
		ends_with,
		input_size,
		batch_size):
        
    	self.default = DEFAULT_DATA_PATH
    	self.offsets = offsets
    	self.sequence_size = sequence_size
    	self.shuffle_series = shuffle_series
    	self.random_State = random_State
    	self.ends_with = ends_with
    	self.input_size = input_size
    	self.batch_size = batch_size


    
    def __get_input(self, path, target_size):
    
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, path, num_classes):
    	y_steer = int(path.split('_')[1])
    	y_throttle = int(path.split('_')[2])
    	return (y_steer,y_throttle)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples


        return X_batch, Y_batch
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size


#Implement Series processing
#Implement Inference







DEFAULT_DATA_PATH = ""
SEED = datetime.time.now()

if use sequences:
#Use randomized 10200 sequences
	Samples = data_gen.get_samples_in_sequence(
	DEFAULT_DATA_PATH, 
	offsets=offsets, 
	sequence_size=sequence_size, 
	shuffle_series=shuffle_data,
	random_State=SEED, 
	ends_with=ends_with)

else:
# Use randomized images
	_, samples = data_gen.get_sample_list(
		DEFAULT_DATA_PATH, 
		do_shuffle=shuffle_data, 
		randon_state=SEED,
		ends_withs=ends_with)

if ant_data < 1.0:
# Use only a portion of the entire dataset samples
	train_test_split(samples, random_state=SEED, test_size = ant_data)
if not skip_test:
	# Splitting the training set to training and validation sets
	# using sklearn. .20 Indicates 20% of the training Is used for validation. train_samples, validation_samples |
	
	train_test_split(samples, random_state=SEED, test_size=0.20)
	
	# Splitting the training set into training and test sets
 	# using sklearn. .12 Indicates 124 of the training is test set. 
 	train_samples, test_samples = train_test_split(train_samples,
 													random_state=SEED,
 													test_size=0.12)

	test_generator = data_gen.single_samples_generator(test_samples,
														ends_with=ends_with, 
														color_depth=False, 
														bw_depth=use_depth, 
														batch_size=mission_models.BATCH_SIZE, 
														sample _name="test")

	test_steps = int(len(test_samples) / mission_nodels.BATCH_SIZE)


train_generator = data_gen.single_samples_generator(train_samples,
													color_deptha = false,
													bw_depthmuse_depth,
													batch_size=mission_models.BATCH_SIZE, 
													sample_names"train"
													ends_with=ends_with)

validation_generator = data_gen.single_samples_generator(validation_samples,
														color_depth = False,
														bw_depth = use_depth,
														batch_size = mission_models.BATCH_SIZE,
														sample_name= "validation", 
														ends_with = ends_with)



