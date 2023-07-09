import os
import cv2
import json
import pathlib
import argparse
import itertools
import numpy as np

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='The path to the pictures directory', required=True)
    parser.add_argument('-n', '--number', help='The plate number', required=True)
    parser.add_argument('-d', '--drug', help='The drug name', required=True)
    parser.add_argument('-f', '--format', help='The layout of colonies of the plate (384, 1536)', required=True)
    parser.add_argument('-ct', '--colony_threshold', help='The threshold for the colony detection as the intesity of the grayscale pixel. (0,255)', required=True)
    parser.add_argument('-o', '--is_order_only', help='Only order the pictures', action='store_true')
    parser.add_argument('-qc', help='generate QC pictures', action='store_true')
    
    args = parser.parse_args()
    path = os.path.normcase(args.path)
    plate_num = int(args.number)
    drug_name = args.drug
    plate_format = int(args.format)
    colony_threshold = int(args.colony_threshold)
    is_order_only = args.is_order_only
    is_generate_qc = args.qc
    input_images = get_files_from_directory(path , '.png')
    organized_images = {}

    config = ""
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    colonies_location = config[f"colony_spacing_and_locations_{plate_format}"]
    global start_x
    start_x = colonies_location["start_x"]
    global start_y
    start_y = colonies_location["start_y"]
    global step_x
    step_x = colonies_location["step_x"]
    global step_y
    step_y = colonies_location["step_y"]

    # Get the start points for trimming the raw pictures
    picture_trim_info = config[f"picture_trim_info_{plate_format}"]
    start_row = picture_trim_info["start_row"]
    end_row = picture_trim_info["end_row"]
    start_col = picture_trim_info["start_col"]
    end_col = picture_trim_info["end_col"]

    text_division_of_origin_96_well_plate = config["text_division_of_origin_96_well_plate"]
    numaric_division_of_origin_96_well_plate = config["numaric_division_of_origin_96_well_plate"]
    
    output_dir_images = create_directory(path, f'ISO_PL_{plate_num}_preproccesed_images')
    QC_dir = create_directory(path, f'QC_ISO_PL_{plate_num}')
    
    organized_images = preprocess_images(input_images, start_row, end_row, start_col, end_col, plate_num, drug_name, colony_threshold, output_dir_images)

    # If the user has chosen to only trim and transform the pictures, return
    if is_order_only:
        return

    if(is_generate_qc):
        generate_qc_images(organized_images, QC_dir)

    # Make a list of the wells in the original 96 well plate
    origin_wells = itertools.product(range(8), range(12))
    
    # Get the areas in the experiment plate
    growth_areas = get_growth_areas(plate_format)
    
    # Store the future area lists each under the wells in the original 96 well plate
    template_dict = {text_division_of_origin_96_well_plate[0]: "", text_division_of_origin_96_well_plate[1]: "", text_division_of_origin_96_well_plate[2]: ""}
    exp_24_areas, ND_24_areas, exp_48_areas, ND_48_areas = init_area_containers(plate_format, template_dict)

    exp_24_areas = {organized_images[0]}
    

    for diviosion in text_division_of_origin_96_well_plate:
        # Get the images for the current experiment based on their original wells
        exp_images = group_expriment_images(diviosion, list(organized_images.keys()))

        # Get the growth areas for each well in the original 96 well plate
        for origin_well_row_index, origin_well_column_index in origin_wells:
            exp_growth_areas = list(convert_original_index_to_experiment_wells_indexes(origin_well_row_index, origin_well_column_index, plate_format))
            
            # Get the growth areas for the current well in all plates
            row_slice = slice(exp_growth_areas[0][0],exp_growth_areas[-1][0] + 1)
            column_slice = slice(exp_growth_areas[0][1],exp_growth_areas[-1][-1] + 1)

            well_growth_areas = growth_areas[row_slice, column_slice]
            
            for row_index, row in enumerate(well_growth_areas):
                for column_index, item in enumerate(row):
                    areas = calculate_growth_area(exp_images, item["start_y"], item["end_y"], item["start_x"], item["end_x"])
                                

def create_directory(father_directory, nested_directory_name):
    '''
    Description
    -----------
    Create a directory if it does not exist
    
    Parameters
    ----------
    father_directory : str
        The path to the directory under which the new directory will be created
    nested_directory_name : str
        The name of the nested directory to be created
    '''
    # Create the output directory path
    new_dir_path = os.path.join(father_directory, nested_directory_name)
    # Create the directory if it does not exist
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)
    return new_dir_path


def get_files_from_directory(path , extension):
    '''Get the full path to each file with the extension specified from the path'''
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(path ,file))
    return files


def preprocess_images(input_images, start_row, end_row, start_col, end_col, plate_num, drug_name ,colony_threshold, output_path):
    '''
    Description
    -----------
    Preprocess the images by cropping them, and converting to black and white them saving them in the outpit path directory

    Parameters
    ----------
    input_images : list
        A list of the paths to the images to be preprocessed
    start_row : int
        The row index of the top left corner of the area to be cropped
    end_row : int
        The row index of the bottom right corner of the area to be cropped
    start_col : int
        The column index of the top left corner of the area to be cropped
    end_col : int
        The column index of the bottom right corner of the area to be cropped
    plate_num : int
        The number of the plate from which the images were taken
    drug_name : str
        The name of the drug used in the experiment
    colony_threshold : int
        The threshold used for determining if a pixel is part of a colony or not
    output_path : str
        The path to the directory in which the preprocessed images will be saved

    Returns
    -------
    dict
        A dictionary containing the images after preprocessing under the name of the image without the extension
    '''

    organized_images = {}

    for picture in input_images:
        picture_name = pathlib.Path(picture).stem.lower()

        image = cv2.imread(picture)
        # Crop the image
        # start_row:end_row, start_col:end_col
        cropped_image = image[start_row:end_row, start_col:end_col]
        
        if '1-4' in picture_name:
            numbers = [1, 2, 3, 4]
        elif '5-8' in picture_name:
            numbers = [5, 6, 7, 8]
        elif '9-12' in picture_name:
            numbers = [9, 10, 11, 12]
        else:
            print(f"file {picture_name} does not contain the numbers in the name")
            return ValueError('The numbers are not specified in the picture name in a supported format. 1-4, 5-8 or 9-12 should be in the name')

        # Get the time from the picture name
        time = ''
        # Check if the time is 24 or 48 hours in the picture name
        if any([len(picture_name.split('24hr')) > 1, len(picture_name.split('24hours')) > 1, len(picture_name.split('24 hours')) > 1]):
            time = '24hr'
        elif any([len(picture_name.split('48hr')) > 1, len(picture_name.split('48hours')) > 1, len(picture_name.split('48 hours')) > 1]):
            time = '48hr'
        else:
            print(f"file {picture_name} does not contain the time in the name")
            return ValueError('The time is not specified in the picture name in a supported format. HOURS or hr should be in the name')

        if len(picture_name.split('nd')) > 1:
            current_image_name = f'ISO_PL_{plate_num}_{drug_name}_{time}_{numbers[0]}-{numbers[-1]}_ND'
        else:
            current_image_name = f'ISO_PL_{plate_num}_{drug_name}_{time}_{numbers[0]}-{numbers[-1]}'

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        mask = cropped_image > colony_threshold
        cropped_image[mask] = 255
        cropped_image[~mask] = 0
        organized_images[current_image_name] = cropped_image
        cv2.imwrite(os.path.join(output_path, current_image_name + ".png"), cropped_image)

    return organized_images


def generate_qc_images(organized_images, output_path):
    '''
    Description
    -----------
    
    Parameters
    ----------
    organized_images : dict
        A dictionary containing the images after preprocessing under the name of the image without the extension
    output_path : str
        The path to the directory in which the preprocessed images will be saved
    
    Returns
    -------
    Boolean
        True if the images were saved successfully, False otherwise
    '''

    # Overlay a grid on the images for visualizing the areas in which the calculations are done
    for image_name ,image in organized_images.items():
        height, width = image.shape
        grid_color = 255

        # Draw vertical grid lines
        for i in range(0, 49):
            cv2.line(image, (start_x + round(i * step_x), 0), (start_x + round(i * step_x), height), grid_color, 1)

        # Draw horizontal grid lines
        for i in range(0, 33):
            cv2.line(image, (0, start_y + round(i * step_y)), (width, start_y + round(i * step_y)), grid_color, 1)

        # Save the result image
        cv2.imwrite(f'{output_path}/{image_name}.png', image)
    
    return True 


def init_area_containers(plate_format, template_dict):
    '''
    Description
    -----------
    Initialize the containers for the growth areas with the appropriate dimensions for the plate format
    under the keys as provided in the template dictionary
    
    Parameters
    ----------
    plate_format : int
        The format of the plate. Only 1536 is supported at the moment
    template_dict : dict
        A dictionary with the keys as the names of the areas and the values as empty strings
    
    Returns
    -------
    dicts: exp_24_areas, ND_24_areas, exp_48_areas, ND_48_areas
    with the keys as the names provided in the template dictionary and the values as numpy arrays with the appropriate dimensions

    '''

    exp_24_areas = template_dict.copy()
    ND_24_areas = template_dict.copy()
    exp_48_areas = template_dict.copy()
    ND_48_areas = template_dict.copy()

    if plate_format == 1536:
        for key in template_dict.keys():
            exp_24_areas[key] = np.zeros((32, 48), dtype=np.float32)
            ND_24_areas[key] = np.zeros((32, 48), dtype=np.float32)
            exp_48_areas[key] = np.zeros((32, 48), dtype=np.float32)
            ND_48_areas[key] = np.zeros((32, 48), dtype=np.float32)
    else:
        raise ValueError('The plate format is not supported')
    
    return exp_24_areas, ND_24_areas, exp_48_areas, ND_48_areas


def group_expriment_images(row_txt, paths_to_images):
    '''
    Description
    -----------
    get all the paths to the files containing the images for each experiment by wells

    Parameters
    ----------
    row_txt : str
        The text desrction of the wells in the intial plate layout that the images were generated from
    paths_to_images : list
        A list of the paths to the images that were generated from the initial plate layout after segmentation
    
    Returns
    ----------
    dict with the following structure:
    {
        "24hr_ND" : "image_path_24hr_ND",
        "24hr" : "image_path_24hr",
        "48hr_ND" : "image_path_48hr_ND",
        "48hr" : "image_path_48hr"
    }
    '''
    exp_images = {}
    for i, image_path in enumerate(paths_to_images):
        if row_txt in image_path:
            if "24hr" in image_path and "ND" in image_path:
                exp_images["24hr_ND"] = image_path
            elif "24hr" in image_path:
                exp_images["24hr"] = image_path
            elif "48hr" in image_path and "ND" in image_path:
                exp_images["48hr_ND"] = image_path
            elif "48hr" in image_path:
                exp_images["48hr"] = image_path
    
    return exp_images


def convert_original_index_to_experiment_wells_indexes(origin_well_row, origin_well_column, plate_format):
    '''
    Description
    -----------
    Convert the index from the original plate to the index in the experiment plate
    
    Parameters
    ----------
    origin_row_index : int
        The row index of the current growth area
    origin_column_index : int
        The column index of the current growth area
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
        
    Returns
    -------
    A string with the well from which the growth area cells were taken - for example "A1"
    '''
    if plate_format == 1536:
        # Each row from the original plate is multiplied to 4 rows in the experiment plate
        row_indexes = [origin_well_row + i for i in range(4)]

        # Based on the column index of the original plate map to the column index of the experiment plate
        # Since everythin is indexed on base 0 the column indexes are reduced by 1
        if origin_well_column in [0,4,8]:
            column_indexes = list(range(1,12))
        elif origin_well_column in [1,5,9]:
            column_indexes = list(range(12,24))
        elif origin_well_column in [2,6,10]:
            column_indexes = list(range(25,37))
        elif origin_well_column in [3,7,11]:
            column_indexes = list(range(36,48))

        # All the growth areas that originated from the same well in the original well
        # are given by the product of the row and column indexes as done here
        return itertools.product(row_indexes, column_indexes)

    else:
        raise ValueError(f'Format {plate_format} is not supported')


def get_growth_areas(plate_format):
    '''
    Description
    -----------
    Get the coordinates of the growth areas in the images
    
    Parameters
    ----------
    plate_format : int
        how many growth areas are in the image - how many colonies were plated on the plate (96, 384, 1536) in the arrayed pattern
    
    Returns
    -------
    A two dimentional numpy array indexed as (Row, Column) with each element containing a dictionary with the following structure:
        "start_x" : start_x_value,
        "end_x" : end_x_value,
        "stary_y" : start_x_value,
        "end_y" : end_y_value
    
    The areas are ordered from top to bottom then left to right
    and therefore correspond to the order of growth areas in the plate
    '''
    areas = np.empty((32, 48), dtype=object)

    if plate_format == 1536:
        # A 1536 well plate has 32 rows and 48 columns and therefore generate 1536 areas
        # first itarate over the rows
        for row_index in range(32):
            # then iterate over the columns producing the areas in the order of 
            # left to right and top to bottom as required
            for column_index in range(48):
                curr_area_start_x = start_x + round(column_index * step_x)
                curr_area_end_x = (start_x + round((column_index + 1) * step_x))

                curr_area_start_y = start_y + round(row_index * step_y)
                curr_area_end_y = (start_y + round((row_index + 1) * step_y))
                
                areas[row_index, column_index] = {
                    "start_x" : curr_area_start_x,
                    "end_x" : curr_area_end_x,
                    "start_y" : curr_area_start_y,
                    "end_y" : curr_area_end_y
                    }
    else:
        raise ValueError(f'Format {plate_format} is not supported')
    
    return areas


def calculate_growth_area(organized_images ,start_y, end_y, start_x, end_x):
    '''
    Description
    -----------
    Calculate the growth within an area in the images provided in the dictionary
    
    Parameters
    ----------
    organized_images : dict
        Contains the the images as a dataframe under the keys as provided in the group_expriment_images function
    start_x : int
        The start x coordinate of the area
    end_x : int
        The end x coordinate of the area
    start_y : int
        The start y coordinate of the area
    end_y : int
        The end y coordinate of the area
    
    Returns
    ----------
    dictionary with the keys as given in the images_dict and the values as the growth area within the area provided
    '''
    growth_areas = {}
    for key, image in organized_images.items():
        growth_areas[key] = image.iloc[start_y:end_y, start_x:end_x].count()
    return growth_areas


if __name__ == "__main__":
    main()