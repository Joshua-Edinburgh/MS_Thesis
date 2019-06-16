import pickle
import random
from PIL import Image
import numpy as np

colors = {'red': [1, 0, 0], 'blue': [0, 0, 1], 'green': [0, 1, 0], 'white': [1]*3, 'gray': [0.5]*3,
          'yellow': [1, 1, 0.1], 'cyan': [0, 1, 1], 'magenta': [1, 0, 1]}
colors_list = list(colors.keys())
object_types = ['box', 'sphere', 'cylinder', 'torus', 'ellipsoid']



def gen_objects(types = 'all'):
    '''
        Yield a list of objects in the form of tuple: (color, type)
        Three possible types: all, test, train
    '''
    # ========== Generate all possible objects ========
    ob_list = []
    for ky1 in colors.keys():
        for ky2 in object_types:
            ob_list.append((ky1,ky2))
    
    # ========== Generate indexes for test ========
    test_indexs = []
    test_object = []
    for i in range(5):
        tmp_test = random.randint(0,7)
        if tmp_test in test_indexs:
            while (tmp_test in test_indexs):
                tmp_test = random.randint(0,7)
        test_indexs.append(tmp_test)
        test_object.append((colors_list[tmp_test],object_types[i]))
    
    if types == 'all':   
        return ob_list
    elif types == 'test':
        return test_object
    elif types == 'train':
        for i in range(len(test_object)):
            ob_list.remove(test_object[i])
        return ob_list
    else: 
        print('types for this function should be "train, test, or all" ')
        return ob_list



def load_images_dict(data_n_samples):
    cache_filename = 'assets/cache.pkl'

    print("Looking for cache file %s" % cache_filename)
    try:
        images_cache = pickle.load(open(cache_filename, 'rb'))
        return images_cache
    except FileNotFoundError:
        print('No cache file, trying to create one...')
    except Exception as e:
        print('Error loading cache file', e)
        exit()

    images_cache = {}
    for color in colors:
        for object_type in object_types:
            for i in range(0, data_n_samples):
                path = 'assets/%s-%s-%d.png' % (color, object_type, i)
                images_cache[color, object_type, i] = np.array(
                    list(Image.open(path).getdata())).reshape((64, 64, 3))

    pickle.dump(images_cache, open('assets/cache.pkl', 'wb'))
    print("Saved cache file %s" % cache_filename)

    return images_cache


def pick_random_color(exclude=None):
    available_colors = list(colors.keys())
    if exclude is not None:
        available_colors.remove(exclude)

    return random.choice(available_colors)


def pick_random_object_type(exclude=None):
    available_object_types = list(object_types)
    if exclude is not None:
        available_object_types.remove(exclude)

    return random.choice(available_object_types)

def check_and_repick(picked, test_list, exclude_color=None, exclude_type=None):
    '''
        Check whether the picked object in the test set, if so, repick
    '''
    while (picked in test_list):
        if exclude_color != None:       # Then we should have the same shape
            new_color = pick_random_color(exclude_color)
            new_shape = picked[1]
            picked = (new_color, new_shape)
        elif exclude_type != None:     # Then we should have the same color
            new_color = picked[0]
            new_shape = pick_random_object_type(exclude_type)
            picked = (new_color, new_shape)
        else:            
            picked = pick_random_color(exclude_color), pick_random_object_type(exclude_type)
        
    return picked
    

def get_batches(images_cache, data_n_samples, n_batches=20, batch_size=50):
    '''
        In each batch, we have four different pairs:
            1. same color same shape
            2. same color diff shape
            3. diff color same shape
            4. rand color rand shape
        Note that the same color and same shape can have different positions.
        Output structure:
            
    '''
    test_list = gen_objects(types = 'test')
    batches = []
    
    partial_same = random.random()*0.5+0.2
    n_same = int(partial_same*batch_size)
    n_same_shape = int(0.15*batch_size)
    n_same_color = int(0.15*batch_size)
    n_random = batch_size - n_same_shape - n_same_color - n_same

    for ib in range(n_batches):
        pairs = []

        for i in range(n_same):
            color, object_type = pick_random_color(), pick_random_object_type()
            color, object_type = check_and_repick((color, object_type), test_list, None, None)
            pairs.append(((color,object_type), (color, object_type)))
            
            if (color,object_type) in test_list or (color,object_type) in test_list:
                print('!!!!!!!!!!!!001!!!!!!!!!!!!')

        for i in range(n_same_shape):
            color, object_type = pick_random_color(), pick_random_object_type()
            color, object_type = check_and_repick((color, object_type), test_list, None, None)
            color2 = pick_random_color(exclude=color)
            color2, object_type = check_and_repick((color2, object_type), test_list, exclude_color=color)
            pairs.append(((color, object_type), (color2, object_type)))

            if (color,object_type) in test_list or (color2,object_type) in test_list:
                print('!!!!!!!!!!!!002!!!!!!!!!!!!')


        for i in range(n_same_color):
            color, object_type = pick_random_color(), pick_random_object_type()
            color, object_type = check_and_repick((color, object_type), test_list, None, None)
            object_type2 = pick_random_object_type(exclude=object_type)
            color, object_type2 = check_and_repick((color, object_type2), test_list, exclude_type=object_type)
            pairs.append(((color, object_type), (color, object_type2)))

            if (color,object_type) in test_list or (color,object_type2) in test_list:
                print('!!!!!!!!!!!!003!!!!!!!!!!!!')

        for i in range(n_random):
            color, object_type = pick_random_color(), pick_random_object_type()
            color, object_type = check_and_repick((color, object_type), test_list, None, None)
            color2, object_type2 = pick_random_color(), pick_random_object_type()
            color2, object_type2 = check_and_repick((color2, object_type2), test_list, None, None)            
            pairs.append(((color, object_type), (color2, object_type2)))

            if (color,object_type) in test_list or (color2,object_type2) in test_list:
                print('!!!!!!!!!!!!004!!!!!!!!!!!!')



        input1 = []     # Image 1
        input2 = []     # Image 2
        labels = []     # Whether they're the same
        descriptions = [] # Description of two figures, two tuples

        for pair in pairs:
            max_i = data_n_samples
            (color1, object_type1), (color2, object_type2) = pair
            label = object_type1 == object_type2 and color1 == color2

            id1 = random.randint(0, max_i-1)
            img1 = images_cache[color1, object_type1, id1] / 256    # Each pixel should in [0,1]
            
            # Here avoid selecting same figure when label=True
            if label:
                available_ids = list(range(id1)) + list(range(id1+1, max_i))
                id2 = random.choice(available_ids)
            else:
                id2 = random.randint(0, max_i - 1)
            img2 = images_cache[color2, object_type2, id2] / 256

            input1.append(img1)
            input2.append(img2)
            labels.append(int(label))
            descriptions.append(((object_type1, color1), (object_type2, color2)))

        batches.append((np.array(input1), np.array(input2), labels, descriptions))

    return batches
