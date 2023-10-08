import os
import cv2
import json
import pathlib
import argparse
import itertools
import matplotlib
import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt


def main():
    styles = ['science', 'notebook', 'grid']
    plt.style.use(styles)

    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='The path to the pictures directory', required=True)
    parser.add_argument('-pn', '--prefix_name', help='A prefix to add to the name of the output files', default="")
    parser.add_argument('-n', '--number', help='The plate number', default=1)
    parser.add_argument('-f', '--format', help='The layout of colonies of the plate (384, 1536)', required=True)

    parser.add_argument('-m', '--media', help='The growth nedia on which the experiment was done', required=True)
    parser.add_argument('-t', '--temperature', help='The incubation temperature in celsius', required=True)
    parser.add_argument('-d', '--drug', help='The drug name', required=True)


    args = parser.parse_args()
    input_path = os.path.normcase(args.path)
    prefix_name = args.prefix_name
    plate_num = int(args.number)
    plate_format = int(args.format)

    media = args.media
    temprature = args.temperature
    drug_name = args.drug
    
    # Get data from config file
    config = ''
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    colonies_location = config[f'colony_spacing_and_locations_{plate_format}']
    global start_x
    start_x = colonies_location['start_x']
    global start_y
    start_y = colonies_location['start_y']
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
    # Distance of Inhibition cutoff
    DI_cutoff = config["DI"]
    colony_threshold = config["CT"]
    
    # Create the output directories
    output_dir_images = create_directory(input_path, f'ISO_PL_{plate_num}_preproccesed_images')
    QC_dir = create_directory(input_path, f'QC_ISO_PL_{plate_num}')
    QC_individual_wells_dir = create_directory(QC_dir, 'individual_wells')
    # Create the output directories for the processed data
    output_dir_processed_data = create_directory(input_path, f'ISO_PL_{plate_num}_processed_data')
    output_dir_graphs = create_directory(input_path, f'ISO_PL_{plate_num}_graphs')

    save_run_parameters(QC_dir, input_path, prefix_name, media, temprature, plate_num, drug_name, plate_format, colony_threshold)

    
    # This is slightly dirty but it is the easiest and won't require changes if someone wants to use tests
    # Make sure not to have any files in the input directory for this option
    if 'simple_tests' in input_path:
        generate_test_images(input_path, start_row, start_col)


    input_images = get_files_from_directory(input_path , '.png')
    organized_images = {}

    # Get the images from the input directory 
    input_images = get_files_from_directory(input_path , '.png')
    
    # Check which divisions have files present in the input directory
    active_divisions = check_active_divisions(input_images, text_division_of_origin_96_well_plate)


    organized_images, trimmed_images = preprocess_images(input_images, start_row, end_row, start_col, end_col, prefix_name, media, temprature, plate_num, drug_name, colony_threshold, plate_format, output_dir_images)


    # Get the areas in the experiment plates
    growth_area_coordinates = get_growth_areas_coordinates(plate_format)
    # Save growth areas in excel file
    growth_areas_df = pd.DataFrame(growth_area_coordinates.reshape(-1, 1), columns=['growth_areas'])
    growth_areas_df.to_excel(os.path.join(QC_dir, f'ISO_PL_{plate_num}_growth_areas.xlsx'), index=False)


    calculated_areas = {}
    for image_name, image in organized_images.items():
        image_areas = calculate_growth_in_areas(image_name ,image, growth_area_coordinates)
        calculated_areas[image_name] = image_areas[image_name]

    raw_areas_df = organize_raw_data(calculated_areas, plate_format)
    raw_areas_df.to_excel(os.path.join(output_dir_processed_data, f'ISO_PL_{plate_num}_raw_data.xlsx'), index=False)


    # Calculate the DI (Distance of Inhibition) for each strain
    DI_df = create_DI_df(raw_areas_df, plate_format, DI_cutoff, text_division_of_origin_96_well_plate, active_divisions)
    
    FoG_df = create_FoG_df(raw_areas_df, DI_df, plate_format, text_division_of_origin_96_well_plate, active_divisions)

    # Merge the DI (Distance of Inhibition) and FoG dataframes on row_index, column_index
    processed_data_df = pd.merge(DI_df, FoG_df, on=['row_index', 'column_index', 'file_name_24hr'])

    # Add the experiment conditions to the dataframe
    processed_data_df['media'] = media
    processed_data_df['temprature'] = temprature
    processed_data_df['drug'] = drug_name
    processed_data_df['plate_format'] = plate_format

    # Add the origin well text field
    processed_data_df = add_well_text_to_df_from_origin_row_row_and_column(processed_data_df)

    # Set column order to file_name_24hr, file_name_48hr, row_index, column_index, DI, FoG, media, temprature, drug, plate_format
    processed_data_df = processed_data_df[['file_name_24hr', 'file_name_48hr', 'origin_well', 'row_index', 'column_index', 'DI', 'FoG', 'media', 'temprature', 'drug', 'plate_format']]

    processed_data_df.to_excel(os.path.join(output_dir_processed_data, f'ISO_PL_{plate_num}_summary_data.xlsx'), index=False)

    generate_qc_images(organized_images, growth_area_coordinates, raw_areas_df, processed_data_df, text_division_of_origin_96_well_plate, active_divisions, plate_format, QC_dir, QC_individual_wells_dir)

    create_FoG_and_DI_hists(processed_data_df, output_dir_graphs, prefix_name, plate_num, DI_cutoff)

    create_distance_from_strip_vs_colony_size_graphs(trimmed_images, growth_area_coordinates, raw_areas_df, processed_data_df, text_division_of_origin_96_well_plate, output_dir_graphs, plate_format, active_divisions)


def get_files_from_directory(path , extension):
    '''Get the full path to each file with the extension specified from the path'''
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(path ,file))
    return files


def create_directory(parent_directory, nested_directory_name):
    '''
    Description
    -----------
    Create a directory if it does not exist
    
    Parameters
    ----------
    parent_directory : str
        The path to the directory under which the new directory will be created
    nested_directory_name : str
        The name of the nested directory to be created
    '''
    # Create the output directory path
    new_dir_path = os.path.join(parent_directory, nested_directory_name)
    # Create the directory if it does not exist
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)
    return new_dir_path


def save_run_parameters(output_dir_processed_data, input_path, prefix_name, media, temprature, plate_num, drug_name, plate_format, colony_threshold):
    with open(os.path.join(output_dir_processed_data, f'{prefix_name}_{plate_num}_run_parameters.txt'), 'w') as f:
        f.write(f'path: {input_path}\n')
        f.write(f'prefix_name: {prefix_name}\n')
        f.write(f'media: {media}\n')
        f.write(f'temprature: {temprature}\n')
        f.write(f'plate_num: {plate_num}\n')
        f.write(f'drug_name: {drug_name}\n')
        f.write(f'plate_format: {plate_format}\n')
        f.write(f'colony_threshold: {colony_threshold}\n')


def check_active_divisions(input_images, text_division_of_origin_96_well_plate):
    '''
    Description
    -----------
    Check which divisions have files present in the input directory

    Parameters
    ----------
    input_images : list
        A list of the paths to the images to be preprocessed
    text_division_of_origin_96_well_plate : list str
        A list of the text divisions of the original plate layout that the images were generated from ('1-4', '5-8', '9-12')

    Returns
    -------
    active_divisions : dict
        keys :
            is_1-4_active : bool
            is_5-8_active : bool
            is_9-12_active : bool
        indicating if there are files for this division in the input directory
    '''
    active_divisions = {}
    for division in text_division_of_origin_96_well_plate:
        active_divisions[f'is_{division}_active'] = any(division in item for item in input_images)

    return active_divisions


def preprocess_images(input_images, start_row, end_row, start_col, end_col, prefix_name, media, temprature, plate_num, drug_name ,colony_threshold, plate_format, output_path):
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
    prefix_name : str
        A prefix to add to the name of the output files
    media : str
        The growth media used in the experiment
    temprature : int
        The incubation temprature in celsius
    plate_num : int
        The number of the plate from which the images were taken
    drug_name : str
        The name of the drug used in the experiment
    colony_threshold : int
        The threshold used for determining if a pixel is part of a colony or not
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    output_path : str
        The path to the directory in which the preprocessed images will be saved

    Returns
    -------
    dict
        A dictionary containing the images after preprocessing under the name of the image without the extension
    '''

    organized_images = {}
    cropped_images = {}

    for picture in input_images:
        picture_name = pathlib.Path(picture).stem.lower()

        image = cv2.imread(picture)

        # Make sure the image aspect ratio is 4:3 (width:height)
        if image.shape[1] / image.shape[0] != 4/3:
            err_str = f"image {picture_name} is not of aspect ratio 4:3 (width:height)"
            print(err_str)
            return ValueError(err_str)
        

        # Development was done given an image of size 4128x3096
        # Therefore the inputed image needs to be resized to that size
        if image.shape != (3096, 4128, 3):
            image = cv2.resize(image, (4128, 3096))

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

        current_image_name = f'{prefix_name}_{plate_num}_{plate_format}_{media}_{temprature}_{drug_name}_{time}_{numbers[0]}-{numbers[-1]}'
        
        # If the plate has no drug, add ND to the name
        if len(picture_name.split('nd')) > 1:
            current_image_name += "_ND"
        
        # Save the trimmed image
        cropped_images[current_image_name] = cropped_image

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        mask = cropped_image > colony_threshold
        cropped_image[mask] = 255
        cropped_image[~mask] = 0
        organized_images[current_image_name] = cropped_image
        cv2.imwrite(os.path.join(output_path, current_image_name + ".png"), cropped_image)

    return organized_images, cropped_images


def get_growth_areas_coordinates(plate_format):
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
    areas = np.empty((32, 40), dtype=object)

    if plate_format == 1536:
        # A 1536 well plate has 32 rows and 48 columns and therefore generate 1536 areas
        # first itarate over the rows
        for row_index in range(32):

            # Since we skip the rows in which the strip is located, we need to map the row index to the correct row index in the plate
            # for that we define the logical indexes of the columns to keep track where we should add the current growth area
            # it has to be sperated from the actual column index so we can still make progress in the loop despite subtracting 1 from the column index
            logical_column_indexes = 0

            # then iterate over the columns producing the areas in the order of 
            # left to right and top to bottom as required
            for column_index in range(48):

                # Skip the columns in which the strip is located as there are no cells there
                if column_index in [0, 1, 22, 23, 24, 25, 46, 47]:
                    continue

                curr_area_start_x = start_x + round(column_index * step_x)
                curr_area_end_x = (start_x + round((column_index + 1) * step_x))

                curr_area_start_y = start_y + round(row_index * step_y)
                curr_area_end_y = (start_y + round((row_index + 1) * step_y))
                
                areas[row_index, logical_column_indexes] = {
                    "start_x" : curr_area_start_x,
                    "end_x" : curr_area_end_x,
                    "start_y" : curr_area_start_y,
                    "end_y" : curr_area_end_y
                    }
                
                logical_column_indexes += 1
    else:
        raise ValueError(f'Format {plate_format} is not supported')
    
    return areas


def calculate_growth_in_areas(image_name, image, growth_areas):
    '''
    Description
    -----------
    Calculate the growth within all the areas in the image provided
    
    Parameters
    ----------
    image_name : str
        The name of the image
    images : numpy array
        The image in which the growth areas are to be calculated
    growth_areas : numpy array
        Contains the coordinates of the growth areas in the images
    
    Returns
    ----------
    key value pair of the image name and a dictionary with the growth areas as keys and the growth area as values
    '''

    area_sizes = {}
    for row_index in range(0, len(growth_areas)):
        for column_index, growth_area_coordinates_item in enumerate(growth_areas[row_index]):
            start_x = growth_area_coordinates_item["start_x"]
            end_x = growth_area_coordinates_item["end_x"]
            start_y = growth_area_coordinates_item["start_y"]
            end_y = growth_area_coordinates_item["end_y"]

            growth_area_pixels = image[start_y : end_y, start_x : end_x]
            # There are only 0 or 255 as values in the image, therefore nonzero count will be the number of pixels with value 255
            # and that is the area of the growth area as set by the threshold earlier
            area_sizes[row_index, column_index] = np.count_nonzero(growth_area_pixels)

    return {image_name : area_sizes}


def organize_raw_data(calculated_areas, plate_format):
    '''
    Description
    -----------
    Organize the calculated areas into a pandas dataframe
    Fields:
        - image_name : str
        - row_index : int
        - column_index : int
        - distance_from_strip : int
        - area : float

    Parameters
    ----------
    calculated_areas : dict
        A dictionary with the image names as keys and the calculated areas as values under the row and column indexes as keys
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
        
    Returns
    -------
    pandas dataframe
        A dataframe with the fields as described above
    '''

    # Set up the fields of the dataframe
    file_names = []
    row_indexes = []
    column_indexes = []
    distances_from_strip = []
    areas = []

    for image_name in calculated_areas:
        for (row_index, column_index), area in calculated_areas[image_name].items():
            curr_distance_from_strip = get_distance_from_strip(column_index, plate_format)
            
            distances_from_strip.append(curr_distance_from_strip)
            areas.append(area)
            file_names.append(image_name)
            row_indexes.append(row_index)
            column_indexes.append(column_index)
            

    return pd.DataFrame({"file_name": file_names, "row_index": row_indexes, "column_index": column_indexes, "distance_from_strip": distances_from_strip, "area": areas})


def get_distance_from_strip(column_index, plate_format):
    '''
    Description
    -----------
    Get the distance of the growth area from the strip

    Parameters
    ----------
    column_index : int
        The column index of the growth area
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    
    Returns
    -------
    int
        The distance of the growth area from the strip
    '''
    if plate_format == 1536:
        # 0 to 9 (physicaly 2 to 11) are the colonies that are near the leftmost strip and they are going away from it,
        # therefore the distance from the strip is the same as the column_index + 1
        # 0 needs to mapped to 1 and 9 to 10
        if column_index in range(0, 10):
            return column_index + 1
        # 9 to 19 (physicaly 12 to 21) are the colonies that are to the left of the middle strip and they are going away from it,
        # therefore the distance from the strip is the max index (19) minus the column_index
        # 10 needs to be mapped to 10 and 19 to 1
        elif column_index in range(10, 20):
            return 20 - column_index
        
        # In the middle we skip 4 columns that are the strip but this previous steps have already excluded them,
        # therefore the index continues from 20. This is a logical index and will be referred to as such going forward.
        # There are 0 to 39 logical indexes, making the amount of columns 40.
        # The actual count of colonies (whether they are used or not) will be referred to as the physical index, 0 to 47 in this case.
        # Making the amount of physical columns 48.

        # 20 to 30 (physicaly 26 to 35) are the colonies that are to the right of the middle strip and they are going away from it,
        # therefore the distance from the strip is the column_index minus the min index -1 (to make it 1 based)
        # 20 needs to mapped to 1 and 30 to 10
        elif column_index in range(20, 30):
            return column_index - 19
        # 30 to 40 are the colonies that are left of the rightmost strip and they are going away from it,
        # therefore the distance from the strip is the max index 40 minus the column_index
        # 30 needs to mapped to 1 and 40 to 10
        elif column_index in range(30, 40):
            return 40 - column_index
        else:
            raise ValueError(f'Column index {column_index} is not a column in a 1536 well plate, this format has 40 columns within the context of this program')
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def calculate_DI(areas_df,control_plate_24hr_ND_name, experiment_plate_24hr_name, origin_well_row, origin_well_column, plate_format, DI_cutoff):
    '''
    Description
    -----------
    Calculate the DI (Distance of inhibition) a given strain

    Parameters
    ----------
    areas_df : pandas dataframe
        The dataframe containing the areas of the growth areas
    control_plate_24hr_ND_name : str
        The name of the control plate with no drug at 24 hours
    experiment_plate_24hr_name : str
        The name of the experiment plate at 24 hours
    origin_well_row : int
        The row index of the well from which the growth area was taken
    origin_well_column : int
        The column index of the well from which the growth area was taken
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    DI_cutoff : float
        The cutoff for the DI - should be given as a number between 0 and 1

    Returns
    -------
    DI_from_strip : int
        The distance if such an index with the given cutoff exists, -1 otherwise
    '''
    experiment_well_indexes = list(convert_original_index_to_experiment_wells_indexes(origin_well_row, origin_well_column, plate_format))
    ND_df_rows_24hr = get_plate_growth_area_sizes(areas_df, control_plate_24hr_ND_name, experiment_well_indexes, plate_format)
    ND_mean_24hr = ND_df_rows_24hr.area.mean()

    # Find the DI (Distance of Inhibition) by finding the first growth area that has a mean of less than ND_mean * DI_cutoff
    exp_growth_area_sizes = get_plate_growth_area_sizes(areas_df, experiment_plate_24hr_name, experiment_well_indexes, plate_format)

    # We need to reverse the list since we always want the list to have the colonies closest to the strip at the end.
    exp_mean_growth_by_distance_from_strip = exp_growth_area_sizes.groupby('distance_from_strip')['area'].mean().values[::-1]

    # Find the first growth area that has a mean of less than ND_mean * DI_cutoff
    DI_from_strip = -1
    for i, growth_area_size in enumerate(exp_mean_growth_by_distance_from_strip):
        if growth_area_size < ND_mean_24hr * DI_cutoff:
            DI_from_strip = i
            break

    return DI_from_strip


def create_DI_df(areas_df, plate_format, DI_cutoff, text_division_of_origin_96_well_plate, active_divisions):
    '''
    Description
    -----------
    Calculate the DI (Distance of inhibition) for each strain
    
    Parameters
    ----------
    areas_df : pandas dataframe
        The dataframe containing the areas of the growth areas
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    DI_cutoff : int
        The cutoff for the DI (usually 50% growth reduction)
    text_division_of_origin_96_well_plate : list str
        The text division of the original plate layout that the images were generated from (['1-4', '5-8', '9-12'])
    active_divisions : dict
        Indicates which divisions have been used in the experiment

    Returns
    -------
    A pandas dataframe with the following columns:
        file_name : str
            The name of the file from which the growth area was taken
        origin_row_index : int
            The row index of the well from which the growth area was taken
        origin_column_index : int
            The column index of the well from which the growth area was taken
        DI : float
            The DI (Distance of Inhibition) of the strain in the well from which the growth area was taken
    '''
    
    # Create lists to later be used for creating the dataframe
    file_names = []
    origin_row_indexes = []
    origin_column_indexes = []
    DIs = []

    # Make a list of the wells this current experiment plate came from.
    # Each experiment plate contain, at most, 32 strains.
    origin_wells = create_32_well_plate_layout()

    # Get unique file names
    unique_file_names = list(areas_df.file_name.unique())

    if plate_format == 1536:
        # Find the pairs of plates as needed for the calculation of the DI (Distance of Inhibition)
        for division in text_division_of_origin_96_well_plate:
            
            # Skip the inactive divisions
            if not active_divisions[f'is_{division}_active']:
                continue

            plates = get_plates_by_division(division, unique_file_names)
            
            control_plate_24hr_ND = plates['24hr_ND']
            experiment_plate_24hr = plates['24hr']

            for (origin_well_row, origin_well_column) in origin_wells:
                DI_from_strip = calculate_DI(areas_df, control_plate_24hr_ND, experiment_plate_24hr, origin_well_row, origin_well_column, plate_format, DI_cutoff)
        
                file_names.append(experiment_plate_24hr)
                origin_row_indexes.append(origin_well_row)
                origin_column_indexes.append(origin_well_column)
                DIs.append(DI_from_strip)

        # Create the dataframe
        DI_df = pd.DataFrame({'file_name_24hr': file_names,
                                'row_index': origin_row_indexes,
                                'column_index': origin_column_indexes,
                                'DI': DIs})
        return DI_df

          
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def get_plates_by_division(current_division, unique_file_names):
    '''
    Description
    -----------
    Get the plates by division of the original 96 well plate provided

    Parameters
    ----------
    current_division : str
        The text division of the original plate layout that the images were generated from ('1-4', '5-8', '9-12')
    unique_file_names : list
        A list of the unique file names in the dataframe

    Returns
    -------
    experiment_plates : dict
        A dictionary with '24hr', '24hr_ND', '48hr', '48hr_ND' as keys and the file names as values        
    '''
    experiment_plate_24hr = get_plate_name_by_time_ND_and_division(unique_file_names, '24hr', False, current_division)
    control_plate_24hr_ND = get_plate_name_by_time_ND_and_division(unique_file_names, '24hr', True, current_division)
    experiment_plate_48hr = get_plate_name_by_time_ND_and_division(unique_file_names, '48hr', False, current_division)
    control_plate_48hr_ND = get_plate_name_by_time_ND_and_division(unique_file_names, '48hr', True, current_division)
    
    return {'24hr': experiment_plate_24hr[0], '24hr_ND': control_plate_24hr_ND[0],
            '48hr': experiment_plate_48hr[0], '48hr_ND': control_plate_48hr_ND[0]}


def get_plate_name_by_time_ND_and_division(file_names ,time, is_ND, current_division):
    '''
    Helper function to the function get_plates_by_division
    '''
    
    if is_ND:
        files = [file_name for file_name in file_names if (time in file_name) and
                                                     ('ND' in file_name) and
                                                     (current_division in file_name)]
    else:
        files = [file_name for file_name in file_names if (time in file_name) and
                                                     ('ND' not in file_name) and
                                                     (current_division in file_name)]
    return files
        

def create_32_well_plate_layout():
    return list(itertools.product(range(8), range(4)))


def convert_origin_row_to_experiment_rows(origin_row_index, plate_format):
    '''
    Description
    -----------
    convert the origin row index (the row index in the 96 well plate) to the row index in the experiment plate

    Parameters
    ----------
    origin_row_index : int
        The row index of the current growth area in the original plate
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    
    Returns
    -------
    row_indexes : list int
        A list of the row indexes of the growth areas in the experiment plate based on the plate format
    '''

    if plate_format == 1536:
        # Since a 1536 plate has 8 time the rows a 96 well plate, but the rows in the 1536
        # are divided into groups of 4 in our case, we need to multiply the row index by 4 to get the correct row index in the 1536 plate.
        # This allows to filter the dataframe by the correct starting row index
        # and end row is handeled by the function call ahead using the format of the plate
        return [(origin_row_index * 4) + i for i in range(4)]
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def convert_original_index_to_experiment_wells_indexes(origin_row_index, origin_column_index, plate_format):
    '''
    Description
    -----------
    Convert the index from the original plate to the logical indexes in the experiment plate
    
    Parameters
    ----------
    origin_row_index : int
        The row index of the origin well
    origin_column_index : int
        The column index of the origin well
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
        
    Returns
    -------
    A list of tuples containing the row and column indexes of the growth areas in the experiment plate
    '''
    
    experiment_row_indexes = convert_origin_row_to_experiment_rows(origin_row_index, plate_format)
    
    if plate_format == 1536:
        # Based on the column index of the original plate map to the column index of the experiment plate
        # Since everythin is indexed on base 0 the column indexes are reduced by 1
        if origin_column_index == 0:
            column_indexes = list(range(0,10))
        elif origin_column_index == 1:
            column_indexes = list(range(10,20))
        elif origin_column_index == 2:
            column_indexes = list(range(20,30))
        elif origin_column_index == 3:
            column_indexes = list(range(30,40))
        else:
            raise ValueError(f'Column index {origin_column_index} is not an index in the experiment plate')

        # All the growth areas that originated from the same well in the original well
        # are given by the product of the row and column indexes as done here
        return itertools.product(experiment_row_indexes, column_indexes)

    else:
        raise ValueError(f'Format {plate_format} is not supported')


def get_plate_growth_area_sizes(areas_df, plate_name, experiment_well_indexes, plate_format):
    '''
    Description
    -----------
    Get the growth area sizes from the plate that was used in the experiment

    Parameters
    ----------
    areas_df : pandas dataframe
        The dataframe containing the areas of the growth areas
    plate_name : str
        The name of the plate from which the growth areas were taken
    experiment_well_indexes : list
        A list of the indexes of the wells in the experiment plate that were used in the experiment
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    
    Returns
    -------
    A pandas dataframe containing the growth areas from the plate that was used in the experiment
    '''
    if plate_format == 1536:
        return areas_df.loc[(areas_df.file_name == plate_name) &
                                            ((areas_df.row_index >= experiment_well_indexes[0][0]) & 
                                                (areas_df.row_index <= experiment_well_indexes[-1][0])) &
                                            ((areas_df.column_index <= experiment_well_indexes[-1][-1]) &
                                                (areas_df.column_index >= experiment_well_indexes[0][-1])), :]
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def create_FoG_df(areas_df, FoG_df, plate_format, text_division_of_origin_96_well_plate, active_divisions):
    '''
    Description
    -----------
    Calculate the FoG (Fraction of Growth) for each starin in the origin plate

    Parameters
    ----------
    areas_df : pandas dataframe
        The dataframe containing the areas of the growth areas
    DI_df : pandas dataframe
        The dataframe containing the DI (Distances of Inhibition) values of the strains
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    text_division_of_origin_96_well_plate : list str
        The text division of the original plate layout that the images were generated from (['1-4', '5-8', '9-12'])
    active_divisions : dict
        Indicates which divisions have been used in the experiment

    Returns
    -------
    A pandas dataframe with the following columns:
        file_name : str
            The name of the file from which the growth area was taken
        origin_row_index : int
            The row index of the well from which the growth area was taken
        origin_column_index : int
            The column index of the well from which the growth area was taken
        FoG : float
            The fraction of growth of the well from which the growth area was taken
    '''

    # Create lists to later be used for creating the dataframe
    file_names = []
    file_names_24hr = []
    origin_row_indexes = []
    origin_column_indexes = []
    FoGs = []

    # Make a list of the wells this current experiment plate came from.
    # Each experiment plate contain, at most, 32 strains.
    origin_wells = create_32_well_plate_layout()

    # Get unique file names
    unique_file_names = list(areas_df.file_name.unique())
    
    if plate_format == 1536:
        # Find the pairs of plates as needed for the calculation of the DI
        for division in text_division_of_origin_96_well_plate:

            # Skip the inactive divisions
            if not active_divisions[f'is_{division}_active']:
                continue

            plates = get_plates_by_division(division, unique_file_names)

            experiment_plate_24hr = plates['24hr']
            experiment_plate_48hr = plates['48hr']
            control_plate_48hr_ND = plates['48hr_ND']
            
            for (origin_well_row, origin_well_column) in origin_wells:
                # Get the indexes of the growth areas in the experiment plate
                experiment_well_indexes = list(convert_original_index_to_experiment_wells_indexes(origin_well_row, origin_well_column, plate_format))

                # Calculate the mean of the growth areas in the ND plate
                ND_48hr_colony_sizes = get_plate_growth_area_sizes(areas_df, control_plate_48hr_ND, experiment_well_indexes, plate_format)
                ND_mean_48hr = ND_48hr_colony_sizes['area'].mean()

                
                # FoG is defined as the average growth at 48hr in drug over DI divided by the average of growth at 48hr in ND
                # Therefore, get the distance on the DI for the strain from the DI_df and divide the mean area of the colonies 
                # closer to the drug strip (than the DI) by the mean area of the ND

                # Get the specific DI for the strain to filter the relevant growth areas later
                strain_DI = int(FoG_df.loc[(FoG_df.file_name_24hr == experiment_plate_24hr) &
                                        (FoG_df.row_index == origin_well_row) &
                                        (FoG_df.column_index == origin_well_column), 'DI'].values[0])
                

                # if we get a distance of -1 that means that there was no distance at which a reduction of DI_cutoff was reached
                # therefore we skip this well and set FoG to -1 to mark that it was skipped and to be ignored later
                if strain_DI == -1:
                    FoG = -1
                    file_names.append(experiment_plate_48hr)
                    file_names_24hr.append(experiment_plate_24hr)
                    origin_row_indexes.append(origin_well_row)
                    origin_column_indexes.append(origin_well_column)
                    FoGs.append(FoG)
                    continue

                
                # Calculate the mean colony size over the DI
                exp_growth_areas = get_plate_growth_area_sizes(areas_df, experiment_plate_48hr, experiment_well_indexes, plate_format)
                

                exp_mean_colony_size_by_distance_from_strip = exp_growth_areas.groupby('distance_from_strip')['area'].mean().values[::-1]
                
                # Calculate the mean colony size over the DI from all the colonies that are closer to the strip than the DI -
                # Add 1 to the starin DI since we are only calculating the mean colony size that is closer to the strip than the DI
                exp_mean_colony_size_over_DI = exp_mean_colony_size_by_distance_from_strip[strain_DI + 1:].mean()
                
                FoG = exp_mean_colony_size_over_DI / ND_mean_48hr
                
                file_names.append(experiment_plate_48hr)
                file_names_24hr.append(experiment_plate_24hr)
                origin_row_indexes.append(origin_well_row)
                origin_column_indexes.append(origin_well_column)
                FoGs.append(FoG)

        
        FoG_df = pd.DataFrame({'file_name_48hr': file_names,
                                'file_name_24hr': file_names_24hr,
                                'row_index': origin_row_indexes,
                                'column_index': origin_column_indexes,
                                'FoG': FoGs})
        return FoG_df

          
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def add_well_text_to_df_from_origin_row_row_and_column(processed_data_df):
    '''
    Description
    -----------
    Convert the origin row and column indexes to the well text in the original plate

    Parameters
    ----------
    processed_data_df : pandas dataframe
        The dataframe containing the processed data. Must have fields:
            - origin_row_index : int
            - origin_column_index : int
    
    Returns
    -------
    pandas dataframe
        The dataframe with the origin row and column indexes converted to the well text in the original plate
        and added as a column: 'origin_well'
    '''
    # Create a list of the wells in the original 96 well plate
    origin_wells = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    processed_data_df['origin_well'] = processed_data_df.apply(lambda row: f'{origin_wells[row.row_index]}{row.column_index + get_column_offset(row["file_name_24hr"].split("_")[-1])}', axis=1)
    
    return processed_data_df


def convert_origin_row_and_column_to_well_text(origin_row_index, origin_column_index, division):
    # Create a list of the wells in the original 96 well plate
    origin_wells = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    offset = get_column_offset(division)

    return f'{origin_wells[origin_row_index]}{origin_column_index + offset}'


def get_column_offset(division):
    offset = 0
    if division == '1-4':
        offset = 1
    elif division == '5-8':
        offset = 5
    elif division == '9-12':
        offset = 9
    else:
        raise ValueError(f'Division {division} is not supported')
    
    return offset


def get_image_part_for_origin_well(trimmed_images, division, origin_row, origin_column, growth_area_coordinates, plate_format):
    '''
    Description
    -----------
    Get the part of the image that contains the colonies on all experiments plates that came from the given origin well

    Parameters
    ----------
    organized_images : dict
        A dictionary containing the images after preprocessing under the name of the image without the extension
    division : str
        The text division of the original plate layout that the current images were generated from ('1-4', '5-8', '9-12')
    origin_row : int
        The row index of the origin well
    origin_column : int
        The column index of the origin well
    growth_area_coordinates : numpy array
        Contains the coordinates of the growth areas in the images
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
        
    Returns
    -------
    image_parts: dict
        A dictionary with the file names as keys and the image parts as values
        Keys: 24hr, 24hr_ND, 48hr, 48hr_ND
    '''
    
    image_parts = {}

    plates = get_plates_by_division(division, trimmed_images.keys())

    colony_indexes = list(convert_original_index_to_experiment_wells_indexes(origin_row, origin_column, plate_format))

    for plate in plates:
        # image[start_row:end_row, start_col:end_col]
        start_row = growth_area_coordinates[colony_indexes[0][0]][colony_indexes[-1][0]]["start_y"]
        end_row = growth_area_coordinates[colony_indexes[-1][0], colony_indexes[-1][0]]["end_y"]
        start_col = growth_area_coordinates[colony_indexes[0][0], colony_indexes[0][1]]["start_x"]
        end_col = growth_area_coordinates[colony_indexes[-1][0], colony_indexes[-1][1]]["end_x"]
        # Copy to avoid modifying the original image
        image_parts[plate] = trimmed_images[plates[plate]][start_row:end_row, start_col:end_col].copy()

    return image_parts


def make_four_picture_grid(top_left, top_right, bottom_left, bottom_right):
    '''Join 4 images into one image'''
    top = cv2.hconcat([top_left, top_right])
    bottom = cv2.hconcat([bottom_left, bottom_right])
    joined_picutres = cv2.vconcat([top, bottom])
    return joined_picutres


def generate_qc_images(organized_images, growth_area_coordinates, raw_areas_df, processed_data_df, text_division_of_origin_96_well_plate, active_divisions, plate_format, output_path, QC_individual_wells_dir):
    '''
    Description
    -----------
    Draw squares around the growth areas in the images for visualizing the areas in which the calculations are done

    Parameters
    ----------
    organized_images : dict
        A dictionary containing the images after preprocessing under the name of the image without the extension
    growth_area_coordinates : numpy array
        Contains the coordinates of the growth areas in the images
    raw_areas_df : pandas dataframe
        The dataframe containing the areas of the growth areas
    processed_data_df : pandas dataframe
        The dataframe containing the processed data (DI and FoG)
    text_division_of_origin_96_well_plate : list str
        The text division of the original plate layout that the images were generated from (['1-4', '5-8', '9-12'])
    active_divisions : dict
        Indicates which divisions have been used in the experiment
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    output_path : str
        The path to the directory in which the images will be saved
    QC_individual_wells_dir : str
        The path to the directory in which the images of the individual wells will be saved
    
    Returns
    -------
    Boolean
        True if the images were saved successfully, False otherwise
    '''

    # Save the images after the preprocessing to be used for the inidividual growth areas by origin well
    images_with_growth_data = {}

    # Index the raw_areas_df by the file name, row index, and column index
    indexed_raw_areas_df = raw_areas_df.set_index(['file_name', 'row_index', 'column_index'])

    # Overlay a grid on the images for visualizing the areas in which the calculations are done
    for image_name ,original_image in organized_images.items():

        # Copy the image to avoid modifying the original image
        image = original_image.copy()

        
        border_color = (255, 255, 255)
        border_thickness = 2
        
        # Full plate
        for row_index, coordintes_row in enumerate(growth_area_coordinates):
            for column_index, coordinte in enumerate(coordintes_row):

                curr_colony_size = indexed_raw_areas_df.xs((image_name, row_index, column_index), level=['file_name', 'row_index', 'column_index'])['area'].values[0]
                
                start_point = (coordinte["start_x"], coordinte["start_y"])
                end_point = (coordinte["end_x"], coordinte["end_y"])
                image = cv2.rectangle(image, start_point, end_point, border_color, border_thickness)

                # Add the colony size and indexes to the bottom left corner of the rectangle
                cv2.putText(image, f'{curr_colony_size:.0f}', (coordinte["start_x"] + 5, coordinte["start_y"] + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, border_color, 1)

        # Save the result image
        cv2.imwrite(f'{output_path}/{image_name}.png', image)
        images_with_growth_data[image_name] = image

    # Individual growth areas by origin well
    origin_wells = create_32_well_plate_layout()

    for origin_row, origin_column in origin_wells:
        for division in text_division_of_origin_96_well_plate:

            if not active_divisions[f'is_{division}_active']:
                continue

            well_areas = get_image_part_for_origin_well(images_with_growth_data, division, origin_row, origin_column, growth_area_coordinates, plate_format)

            all_areas = make_four_picture_grid(well_areas['24hr_ND'], well_areas['48hr_ND'], well_areas['24hr'], well_areas['48hr'])

            # Save the result image
            cv2.imwrite(f'{QC_individual_wells_dir}/{image_name}_{convert_origin_row_and_column_to_well_text(origin_row, origin_column, division)}.png', all_areas)
    
    return True


def create_hist(data, ax, title, xlabel, linewidth=2):
    '''
    Description
    -----------
    Create a histogram of the data

    Parameters
    ----------
    data : array like
        The data to plot the histogram of
    ax : matplotlib axes object
        The axes on which to plot the histogram
    title : str
        The title of the histogram
    xlabel : str
        The label of the x axis
    linewidth : int
        The width of the line of the histogram    
    
    Returns
    -------
    None
    '''
    # Remove ticks from Y axis
    ax.yaxis.set_ticks_position('none')
    # Remove ticks from X axis top
    ax.xaxis.set_ticks_position('bottom')

    ax.set_title(title)    
    ax.hist(data, bins=10, linewidth = linewidth, histtype='step')
    ax.set_ylabel('Count')
    ax.set_xlabel(xlabel)


def create_FoG_and_DI_hists(processed_data_df, graphs_dir, prefix_name, plate_num, DI_cutoff):
    '''
    Description
    -----------
    Create histograms of the FoG and DI values and save them in the graphs data directory
    
    Parameters
    ----------
    processed_data_df : pandas dataframe
        The dataframe with the FoG and DI values
    graphs_dir : str
        The path to the directory in which the graphs will be saved
    prefix_name : str
        The prefix name of the experiment
    plate_num : int
        The number of the plate from which the images were taken
    DI_cutoff : float
        The cutoff for the DI (usually 0.5 growth reduction)

    Returns
    -------
    None    
    '''
    # Filter out the rows that have FoG of -1
    processed_data_df = processed_data_df.loc[processed_data_df.FoG != -1, :]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    # Create the FoG histogram
    create_hist(processed_data_df.FoG, ax[0], f'FoG distrobution for {prefix_name} {plate_num}', 'FoG')

    # Create the DI histogram
    DI_cutoff_text = f'DI {DI_cutoff * 100:.2f}%'
    create_hist(processed_data_df.DI, ax[1], f'Distance of inhibition (DI) distrobution for {prefix_name} {plate_num}', DI_cutoff_text)
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'{prefix_name}_{plate_num}_FoG_and_{DI_cutoff_text.replace(" " , "_")}_hist.png'), dpi=500)


def create_distance_from_strip_vs_colony_size_graphs(trimmed_images, growth_area_coordinates, raw_areas_df, processed_data_df, text_division_of_96_well_plate, graphs_dir, plate_format, active_divisions):
    '''
    Description
    -----------
    Create graphs of the distance from the strip vs colony size

    Parameters
    ----------
    trimmed_images : dict
        A dictionary containing the images after trimming so only the growth areas remain
    growth_area_coordinates : numpy array
        Contains the coordinates of the growth areas in the images
    raw_areas_df : pandas dataframe
        The dataframe containing the areas of the colonies
    processed_data_df : pandas dataframe
        The dataframe containing the processed data (DI and FoG)
    text_division_of_96_well_plate : list str
        The text division of the original plate layout that the images were generated from (['1-4', '5-8', '9-12'])
    graphs_dir : str
        The path to the directory in which the graphs will be saved
    plate_format : int
        how many growth areas are in the image (96, 384, 1536)
    active_divisions : dict
        Indicates which divisions have been used in the experiment
    
    Returns
    -------
    None
    '''

    # Use Agg backend to avoid displaying the figures and lower RAM usage
    matplotlib.use("Agg")

    origin_wells = create_32_well_plate_layout()

    colors = ['#E63B60', '#067FD0', '#223BC9', '#151A7B']

    for origin_row, origin_column in origin_wells:
        for division in text_division_of_96_well_plate:
            
            if not active_divisions[f'is_{division}_active']:
                continue

            plates = get_plates_by_division(division, trimmed_images.keys())
            well_areas = get_image_part_for_origin_well(trimmed_images, division, origin_row, origin_column, growth_area_coordinates, plate_format)
            area_coordinates_indexes = list(convert_original_index_to_experiment_wells_indexes(origin_row, origin_column, plate_format))

            # Check if need to flip image horizontally so it will be in the same orientation as the other images - strip on the right 
            if origin_column in [0, 2, 4, 6, 8, 10]:
                well_areas['24hr'] = cv2.flip(well_areas['24hr'], 1)
                well_areas['48hr'] = cv2.flip(well_areas['48hr'], 1)

            well_areas['24hr'] = cv2.cvtColor(well_areas['24hr'], cv2.COLOR_BGR2RGB)
            well_areas['48hr'] = cv2.cvtColor(well_areas['48hr'], cv2.COLOR_BGR2RGB)

            # Prep the 24hr raw colony sizes by distance from strip for plotting
            colony_sizes_24hr = get_plate_growth_area_sizes(raw_areas_df, plates['24hr'],
                                                                area_coordinates_indexes, plate_format).groupby('row_index')
            
            colony_sizes_48hr = get_plate_growth_area_sizes(raw_areas_df, plates['48hr'],
                                                                area_coordinates_indexes, plate_format).groupby('row_index')


            # Get the DI 80, 50 and 20 for the current strain
            # 80% inhibition - need to find the area at which the colony size is 20% of the original size. Therefore provide 1 - %inihibition as the DI cutoff
            strain_DI_80 = calculate_DI(raw_areas_df, plates['24hr_ND'], plates['24hr'], origin_row, origin_column, plate_format, 0.2)
                                
            
            # 50% inhibition - need to find the area at which the colony size is 50% of the original size. Therefore provide 1 - %inihibition as the DI cutoff
            strain_DI_50 = int(processed_data_df.loc[(processed_data_df.file_name_24hr == plates['24hr']) &
                                        (processed_data_df.row_index == origin_row) &
                                        (processed_data_df.column_index == origin_column), 'DI'].values[0])

            # 20% inhibition - need to find the area at which the colony size is 80% of the original size. Therefore provide 1 - %inihibition as the DI cutoff
            strain_DI_20 = calculate_DI(raw_areas_df, plates['24hr_ND'], plates['24hr'], origin_row, origin_column, plate_format, 0.8)


            fig, ax = plt.subplots(2, 2, figsize=(15, 9))
            # Set the title of the graph
            fig.suptitle(f'Colony size vs distance from strip for {convert_origin_row_and_column_to_well_text(origin_row, origin_column, division)} plate {division}', fontsize=16)

            ax[0, 0].set_title('24hr')
            ax[0, 0].set_ylabel('Colony size [pixels]')
            ax[0, 0].set_xlabel('Distance from strip')
            ax[0, 0].yaxis.set_ticks_position('left')
            ax[0, 0].xaxis.set_ticks_position('bottom')
            

            ax[0, 1].set_title('48hr')
            ax[0, 1].set_ylabel('Colony size [pixels]')
            ax[0, 1].set_xlabel('Distance from strip')
            ax[0, 1].yaxis.set_ticks_position('left')
            ax[0, 1].xaxis.set_ticks_position('bottom')

            ax[1, 0].set_title('24hr all colonies')
            # Remove grid lines, ticks, tick labels and all spines
            ax[1, 0].grid(False)
            ax[1, 0].set_xticks([])
            ax[1, 0].set_yticks([])

            ax[1, 1].set_title('48hr all colonies')
            ax[1, 1].grid(False)
            ax[1, 1].set_xticks([])
            ax[1, 1].set_yticks([])


            for row_index, row_data in enumerate(colony_sizes_24hr):
                ax[0,0].plot(row_data[1]['distance_from_strip'][::-1], row_data[1]['area'], label=f'Row {row_index + 1}', color=colors[row_index])

            
            for row_index, row_data in enumerate(colony_sizes_48hr):
                ax[0,1].plot(row_data[1]['distance_from_strip'][::-1], row_data[1]['area'], label=f'Row {row_index + 1}', color=colors[row_index])

            # add DIs to plots
            for row_index, column_index in [(0, 0), (0, 1)]:
                if strain_DI_20 != -1:
                    ax[row_index, column_index].axvline(x=strain_DI_20, color='#FA8072', linestyle='--', label=f'DI_20%={strain_DI_20}')
                if strain_DI_50 != -1:
                    ax[row_index, column_index].axvline(x=strain_DI_50, color='#B22222', linestyle=':', label=f'DI_50%={strain_DI_50}')
                if strain_DI_80 != -1:
                    ax[row_index, column_index].axvline(x=strain_DI_80, color='#7C0A02', linestyle='-.', label=f'DI_80%={strain_DI_80}')
                
            ax[1, 0].imshow(well_areas['24hr'])
            ax[1, 1].imshow(well_areas['48hr'])

            legend_transperancy = 0.8
            ax[0, 0].legend(loc='upper right', framealpha=legend_transperancy)
            ax[0, 1].legend(loc='upper right', framealpha=legend_transperancy)

            plt.savefig(os.path.join(graphs_dir, f'{convert_origin_row_and_column_to_well_text(origin_row, origin_column, division)}_distance_vs_colony_size.png'), dpi=200)
            plt.close('all')


def generate_test_images(save_path, start_row, start_column):
    ''' 
    Description
    -----------
    Generate test images

    Parameters
    ----------
    save_path : str
        The path to the directory in which the images will be saved
    start_row : int
        The row index of the first growth area in the experiment plate
    start_column : int
        The column index of the first growth area in the experiment plate

    Returns
    -------
    None
    '''

    # TODO - this is pretty gross at the moment due to the time rush, need to clean it up
    white = (255, 255, 255)
    move_x = 54
    move_y = 54
    y_start_loc = 20 + start_row

    y_start_indexes = []
    for i in range(1, 21):
        y_start_indexes.append(y_start_loc)
        y_start_loc += move_y
    
    # 24hr and 24hr_ND
    img_24hr = np.zeros((3096, 4128, 3), np.uint8)

    # Input actual pixel values for the colony sizes
    colony_sizes = [1, 1, 1, 1, 9, 9, 9, 9, 10, 10]
    for y_start_index in y_start_indexes:
        
        x_location = 152 + start_column
        colony_size = 1
        
        for colony_size in colony_sizes:
            img_24hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
            x_location += move_x

    # A2
    colony_sizes = [10, 10, 9, 9, 9, 9, 1, 1, 1, 1]
    for y_start_index in y_start_indexes[0:4]:
            
            x_location = 152 + start_column + (move_x * 10)
            
            for colony_size in colony_sizes:
                img_24hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                x_location += move_x

    # A3
    colony_sizes = [1, 1, 1, 1, 9, 9, 9, 9, 10, 10]
    for y_start_index in y_start_indexes[0:4]:
                
                x_location = 152 + start_column + (move_x * 24)
                
                for colony_size in colony_sizes:
                    img_24hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                    x_location += move_x


    # A4
    colony_sizes = [10, 10, 9, 9, 9, 9, 1, 1, 1, 1]
    for y_start_index in y_start_indexes[0:4]:
                    
                    x_location = 152 + start_column + (move_x * 34)
                    
                    for colony_size in colony_sizes:
                        img_24hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                        x_location += move_x

    cv2.imwrite(os.path.join(save_path ,'test_24hr_1-4.png'), img_24hr)
    cv2.imwrite(os.path.join(save_path ,'test_24hr_1-4_ND.png'), img_24hr)
    cv2.imwrite(os.path.join(save_path ,'test_24hr_5-8.png'), img_24hr)
    cv2.imwrite(os.path.join(save_path ,'test_24hr_5-8_ND.png'), img_24hr)
    cv2.imwrite(os.path.join(save_path ,'test_24hr_9-12.png'), img_24hr)
    cv2.imwrite(os.path.join(save_path ,'test_24hr_9-12_ND.png'), img_24hr)


    # 48hr ND image
    img_48hr_ND = np.zeros((3096, 4128, 3), np.uint8)
    
    # A1, B1, C1, D1, E1 - all 10
    for y_start_index in y_start_indexes:
            
            x_location = 152 + start_column
            colony_size = 10
            
            for i in range(1, 11):
                img_48hr_ND[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                x_location += move_x


    # A2
    colony_sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[0:4]:
                    
        x_location = 152 + start_column + (move_x * 10)
        
        for colony_size in colony_sizes:
            img_48hr_ND[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
            x_location += move_x


    # A3
    colony_sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[0:4]:
                            
        x_location = 152 + start_column + (move_x * 24)
        
        for colony_size in colony_sizes:
            img_48hr_ND[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
            x_location += move_x

    # A4
    colony_sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[0:4]:
                                        
        x_location = 152 + start_column + (move_x * 34)
        
        for colony_size in colony_sizes:
            img_48hr_ND[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
            x_location += move_x

    cv2.imwrite(os.path.join(save_path ,'test_48hr_1-4_ND.png'), img_48hr_ND)
    cv2.imwrite(os.path.join(save_path ,'test_48hr_5-8_ND.png'), img_48hr_ND)
    cv2.imwrite(os.path.join(save_path ,'test_48hr_9-12_ND.png'), img_48hr_ND)


    # 48hr image
    img_48hr = np.zeros((3096, 4128, 3), np.uint8)

    # A1 - all 10
    colony_sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[0:4]:
            
            x_location = 152 + start_column
            
            for colony_size in colony_sizes:
                img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                x_location += move_x

    # A2 - 10, 10, 10, 10, 10, 10, 10, 2, 2, 2
    colony_sizes = [10, 10, 10, 10, 10, 10, 10, 2, 2, 2]
    for y_start_index in y_start_indexes[0:4]:
                
                x_location = 152 + start_column + (move_x * 10)
                
                for colony_size in colony_sizes:
                    img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                    x_location += move_x

    # A3 - 3, 3, 3, 10, 10, 10, 10, 10, 10, 10
    colony_sizes = [3, 3, 3, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[0:4]:
                
                x_location = 152 + start_column + (move_x * 24)
                
                for colony_size in colony_sizes:
                    img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                    x_location += move_x

    # A4 - 10, 10, 10, 10, 10, 10, 10, 4, 4, 4
    colony_sizes = [10, 10, 10, 10, 10, 10, 10, 4, 4, 4]
    for y_start_index in y_start_indexes[0:4]:
                
                x_location = 152 + start_column + (move_x * 34)
                
                for colony_size in colony_sizes:
                    img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                    x_location += move_x

    # B1- 8, 9, 9, 10, 10, 10, 10, 10. 10, 10
    colony_sizes = [8, 9, 9, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[4:8]:
            
            x_location = 152 + start_column
            
            for colony_size in colony_sizes:
                img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                x_location += move_x 

    # C1 - 4, 5, 6, 10, 10, 10, 10, 10, 10, 10
    colony_sizes = [4, 5, 6, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[8:12]:
            
            x_location = 152 + start_column
            
            for colony_size in colony_sizes:
                img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                x_location += move_x
            
    # D1 - 2, 3, 4, 10, 10, 10, 10, 10, 10, 10
    colony_sizes = [2, 3, 4, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[12:16]:
            
            x_location = 152 + start_column
            
            for colony_size in colony_sizes:
                img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                x_location += move_x

    # E1 - 0, 0, 0, 10, 10, 10, 10, 10, 10, 10
    colony_sizes = [0, 0, 0, 10, 10, 10, 10, 10, 10, 10]
    for y_start_index in y_start_indexes[16:20]:
            
            x_location = 152 + start_column
            
            for colony_size in colony_sizes:
                img_48hr[y_start_index : y_start_index + 1, x_location : x_location + colony_size] = white
                x_location += move_x


    cv2.imwrite(os.path.join(save_path ,'test_48hr_1-4.png'), img_48hr)
    cv2.imwrite(os.path.join(save_path ,'test_48hr_5-8.png'), img_48hr)
    cv2.imwrite(os.path.join(save_path ,'test_48hr_9-12.png'), img_48hr)


if __name__ == "__main__":
    main()