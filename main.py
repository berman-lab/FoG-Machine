import os
import cv2
import json
import pathlib
import argparse
import itertools
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
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
    

    input_images = get_files_from_directory(input_path , '.png')
    organized_images = {}

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
    # Create the output directories for the processed data
    output_dir_processed_data = create_directory(input_path, f'ISO_PL_{plate_num}_processed_data')

    output_dir_graphs = create_directory(input_path, f'ISO_PL_{plate_num}_graphs')

    save_run_parameters(QC_dir, input_path, prefix_name, media, temprature, plate_num, drug_name, plate_format, colony_threshold)
    
    organized_images = preprocess_images(input_images, start_row, end_row, start_col, end_col, prefix_name, media, temprature, plate_num, drug_name, colony_threshold, plate_format, output_dir_images)

    # Get the areas in the experiment plates
    growth_area_coordinates = get_growth_areas_coordinates(plate_format)

    generate_qc_images(organized_images, growth_area_coordinates, QC_dir)

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
    DI_df = calculate_DI(raw_areas_df, plate_format, DI_cutoff, text_division_of_origin_96_well_plate)
    
    FoG_df = calculate_FoG(raw_areas_df, DI_df, plate_format, text_division_of_origin_96_well_plate)

    # Merge the DI (Distance of Inhibition) and FoG dataframes on row_index, column_index
    processed_data_df = pd.merge(DI_df, FoG_df, on=['row_index', 'column_index'])

    # Add the experiment conditions to the dataframe
    processed_data_df['media'] = media
    processed_data_df['temprature'] = temprature
    processed_data_df['drug'] = drug_name
    processed_data_df['plate_format'] = plate_format

    # Set column order to file_name_24hr, file_name_48hr, row_index, column_index, DI, FoG, media, temprature, drug, plate_format
    processed_data_df = processed_data_df[['file_name_24hr', 'file_name_48hr', 'row_index', 'column_index', 'DI', 'FoG', 'media', 'temprature', 'drug', 'plate_format']]

    processed_data_df.to_excel(os.path.join(output_dir_processed_data, f'ISO_PL_{plate_num}_summary_data.xlsx'), index=False)

    create_FoG_and_DI_hists(processed_data_df, output_dir_graphs, prefix_name, plate_num, DI_cutoff)

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
        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        mask = cropped_image > colony_threshold
        cropped_image[mask] = 255
        cropped_image[~mask] = 0
        organized_images[current_image_name] = cropped_image
        cv2.imwrite(os.path.join(output_path, current_image_name + ".png"), cropped_image)

    return organized_images


def generate_qc_images(organized_images, growth_area_coordinates, output_path):
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
    output_path : str
        The path to the directory in which the preprocessed images will be saved
    
    Returns
    -------
    Boolean
        True if the images were saved successfully, False otherwise
    '''

    # Overlay a grid on the images for visualizing the areas in which the calculations are done
    for image_name ,image in organized_images.items():

        # Copy the image to avoid modifying the original image
        image = image.copy()

        
        border_color = (255, 255, 255)
        border_thickness = 2

        # Draw the borders of the growth areas
        for coordintes_row in growth_area_coordinates:
            for coordinte in coordintes_row:
                start_point = (coordinte["start_x"], coordinte["start_y"])
                end_point = (coordinte["end_x"], coordinte["end_y"])
                image = cv2.rectangle(image, start_point, end_point, border_color, border_thickness)

        # Save the result image
        cv2.imwrite(f'{output_path}/{image_name}.png', image)
    
    return True 


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
            # If the current growth area is the strip itself, skip it
            if curr_distance_from_strip == -1:
                continue

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
        # The strip itself is at columns: 0, 1, 22, 23, 24, 25, 46, 47

        # The colonies that are near the leftmost strip and they are going away from it.
        # Therefore the distance from the strip is the same as the column_index -1 
        # column index 2 needs to be mapped to 1
        if column_index in range(2, 12):
            return column_index - 1
        # The colonies that are to the left of the middle strip and they are going away from it
        # Therefore the distance from the strip is the max index (23) minus the column_index
        elif column_index in range(12, 22):
            return 22 - column_index
        # The colonies that are to the right of the middle strip and they are going away from it
        # Therefore the distance from the strip is the column_index minus the min index 25 (26 needs to mapped to 1)
        elif column_index in range(26, 36):
            return column_index - 25
        # The colonies that are left of the rightmost strip and they are going away from it
        # Therefore the distance from the strip is the max index 46 minus the column_index (45 needs to mapped to 1)
        elif column_index in range(36, 46):
            return 46 - column_index
        else:
            return -1        
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def calculate_DI(areas_df, plate_format, DI_cutoff, text_division_of_origin_96_well_plate):
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
    text_division_of_origin_96_well_plate : str
        The text division of the original plate layout that the images were generated from.
        Usually ['1-4', '5-8', '9-12']

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

    # Make a list of the wells in the original 96 well plate
    origin_wells = create_96_well_plate_layout()

    # Get unique file names
    unique_file_names = list(areas_df.file_name.unique())

    # Take the name that has 24hr in it and does not have ND in it
    experiment_plates_24hr = [file_name for file_name in unique_file_names if "24hr" in file_name and "ND" not in file_name]
    # Take the name that has 24hr in it and has ND in it
    control_plates_24hr_ND = [file_name for file_name in unique_file_names if "24hr" in file_name and "ND" in file_name]
    

    if plate_format == 1536:
        # Find the pairs of plates as needed for the calculation of the DI (Distance of Inhibition)
        for division in text_division_of_origin_96_well_plate:
            experiment_plate_24hr = [plate for plate in experiment_plates_24hr if division in plate]
            control_plate_24hr_ND = [plate for plate in control_plates_24hr_ND if division in plate]
            
            # If either are empty that means that the expriement was not done for less strains
            if len(control_plate_24hr_ND) == 0 or len(experiment_plate_24hr) == 0:
                continue
            elif len(control_plate_24hr_ND) > 1 or len(experiment_plate_24hr) > 1:
                raise ValueError(f'There is more than one plate for {division}')

            control_plate_24hr_ND = control_plate_24hr_ND[0]
            experiment_plate_24hr = experiment_plate_24hr[0]

            for (origin_well_row, origin_well_column) in origin_wells:
                # Get the indexes of the growth areas in the experiment plate
                experiment_well_indexes = list(convert_original_index_to_experiment_wells_indexes(origin_well_row, origin_well_column, plate_format))
                
                # Since a 1536 plate has 8 time the rows a 96 well plate, but the rows in the 1536
                # are divided into groups of 4 in our case, we need to multiply the row index by 4 to get the correct row index in the 1536 plate.
                # This allows to filter the dataframe by the correct starting row index
                # and end row is handeled by the function call ahead using the format of the plate
                exp_row = origin_well_row * 4

                ND_df_rows = get_plate_growth_area_sizes(areas_df, control_plate_24hr_ND, exp_row, experiment_well_indexes, plate_format)
                ND_mean = ND_df_rows.area.mean()

                # Find the DI (Distance of Inhibition) by finding the first growth area that has a mean of less than ND_mean * DI_cutoff
                exp_growth_areas = get_plate_growth_area_sizes(areas_df, experiment_plate_24hr, exp_row, experiment_well_indexes, plate_format)
                
                exp_mean_growth_by_distance_from_strip = exp_growth_areas.groupby('distance_from_strip')['area'].mean().values

                # Find the first growth area that has a mean of less than ND_mean * DI_cutoff
                DI_distance_from_strip = -1
                for i, growth_area in enumerate(exp_mean_growth_by_distance_from_strip[::-1]):
                    if growth_area < ND_mean * DI_cutoff:
                        DI_distance_from_strip = len(exp_mean_growth_by_distance_from_strip) - i
                        break
                
                # Add the file name, origin row and column indexes, and the DI
                file_names.append(experiment_plate_24hr)
                origin_row_indexes.append(origin_well_row)
                origin_column_indexes.append(origin_well_column)
                DIs.append(DI_distance_from_strip)

        # Create the dataframe
        DI_df = pd.DataFrame({'file_name_24hr': file_names,
                                'row_index': origin_row_indexes,
                                'column_index': origin_column_indexes,
                                'DI': DIs})
        return DI_df

          
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def create_96_well_plate_layout():
    return list(itertools.product(range(8), range(12)))


def convert_original_index_to_experiment_wells_indexes(origin_well_row, origin_well_column, plate_format):
    '''
    Description
    -----------
    Convert the index from the original plate to the indexes in the experiment plate
    
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
            column_indexes = list(range(2,12))
        elif origin_well_column in [1,5,9]:
            column_indexes = list(range(12,22))
        elif origin_well_column in [2,6,10]:
            column_indexes = list(range(26,36))
        elif origin_well_column in [3,7,11]:
            column_indexes = list(range(36,46))

        # All the growth areas that originated from the same well in the original well
        # are given by the product of the row and column indexes as done here
        return itertools.product(row_indexes, column_indexes)

    else:
        raise ValueError(f'Format {plate_format} is not supported')


def get_plate_growth_area_sizes(areas_df, plate_name, experiment_begin_row, experiment_well_indexes, plate_format):
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
    experiment_begin_row : int
        The row index of the first growth area in the experiment plate
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
                                            ((areas_df.row_index >= experiment_begin_row) & 
                                                (areas_df.row_index <= experiment_begin_row + 3)) &
                                            ((areas_df.column_index <= experiment_well_indexes[-1][-1]) &
                                                (areas_df.column_index >= experiment_well_indexes[0][-1])), :]
    else:
        raise ValueError(f'Format {plate_format} is not supported')


def calculate_FoG(areas_df, DI_df, plate_format, text_division_of_origin_96_well_plate):
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
    The text division of the original plate layout that the images were generated from.
        Usually ['1-4', '5-8', '9-12']

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
    origin_row_indexes = []
    origin_column_indexes = []
    FoGs = []

    # Make a list of the wells in the original 96 well plate
    origin_wells = create_96_well_plate_layout()

    # Get unique file names
    unique_file_names = list(areas_df.file_name.unique())

    # Take the name that has 24hr in it and does not have ND in it to filter the DI_df with
    experiment_plates_24hr = [file_name for file_name in unique_file_names if "24hr" in file_name and "ND" not in file_name]

    # Take the name that has 48hr in it and does not have ND in it
    experiment_plates_48hr = [file_name for file_name in unique_file_names if "48hr" in file_name and "ND" not in file_name]
    # Take the name that has 48hr in it and has ND in it
    control_plates_48hr_ND = [file_name for file_name in unique_file_names if "48hr" in file_name and "ND" in file_name]
    

    if plate_format == 1536:
        # Find the pairs of plates as needed for the calculation of the DI
        for division in text_division_of_origin_96_well_plate:
            experiment_plate_24hr = [plate for plate in experiment_plates_24hr if division in plate]
            experiment_plate_48hr = [plate for plate in experiment_plates_48hr if division in plate]
            control_plate_48hr_ND = [plate for plate in control_plates_48hr_ND if division in plate]

            # If either are empty that means that the expriement was not done for less strains
            if len(control_plate_48hr_ND) == 0 or len(experiment_plate_48hr) == 0 or len(experiment_plate_24hr) == 0:
                continue
            elif len(control_plate_48hr_ND) > 1 or len(experiment_plate_48hr) > 1 or len(experiment_plate_24hr) > 1:
                raise ValueError(f'There are more than one plate for {division}')

            experiment_plate_24hr = experiment_plate_24hr[0]
            experiment_plate_48hr = experiment_plate_48hr[0]
            control_plate_48hr_ND = control_plate_48hr_ND[0]
            
            for (origin_well_row, origin_well_column) in origin_wells:
                # Get the indexes of the growth areas in the experiment plate
                experiment_well_indexes = list(convert_original_index_to_experiment_wells_indexes(origin_well_row, origin_well_column, plate_format))
                
                # Since a 1536 plate has 8 time the rows a 96 well plate, but the rows in the 1536
                # are divided into groups of 4 in our case, we need to multiply the row index by 4
                # to get the correct row index in the 1536 plate
                exp_row = origin_well_row * 4

                # Calculate the mean of the growth areas in the ND plate
                ND_growth_area_sizes = get_plate_growth_area_sizes(areas_df, control_plate_48hr_ND, exp_row, experiment_well_indexes, plate_format)

                ND_mean_48hr = ND_growth_area_sizes['area'].mean()

                
                # FoG is defined as the average growth at 48hr in drug over DI divided by the average of growth at 48hr in ND
                # Therefore, get the distance on the DI for the strain from the DI_df and divide the mean area of the colonies 
                # closer to the drug strip (than the DI) by the mean area of the ND

                # Get the specific DI for the strain to filter the relevant growth areas later
                strain_DI = int(DI_df.loc[(DI_df.file_name_24hr == experiment_plate_24hr) &
                                        (DI_df.row_index == origin_well_row) &
                                        (DI_df.column_index == origin_well_column), 'DI'].values[0])
                

                # if we get a distance of -1 that means that there was no distance at which a reduction of DI_cutoff was reached
                # therefore we skip this well and set FoG to -1 to mark that it was skipped and to be ignored later
                if strain_DI == -1:
                    FoG = -1
                    file_names.append(experiment_plate_48hr)
                    origin_row_indexes.append(origin_well_row)
                    origin_column_indexes.append(origin_well_column)
                    FoGs.append(FoG)
                    continue

                
                # Distance is based 1 counting, so we need to subtract 1 to get the index from which to get the mean of the growth areas
                # that are closer to the drug strip than the DI
                exp_growth_areas = get_plate_growth_area_sizes(areas_df, experiment_plate_48hr, exp_row, experiment_well_indexes, plate_format)
                exp_mean_growth_over_DI = exp_growth_areas.groupby('distance_from_strip')['area'].mean().values[::-1][strain_DI - 1:].mean()
                
                FoG = exp_mean_growth_over_DI / ND_mean_48hr
                
                file_names.append(experiment_plate_48hr)
                origin_row_indexes.append(origin_well_row)
                origin_column_indexes.append(origin_well_column)
                FoGs.append(FoG)

        DI_df = pd.DataFrame({'file_name_48hr': file_names,
                                'row_index': origin_row_indexes,
                                'column_index': origin_column_indexes,
                                'FoG': FoGs})
        return DI_df

          
    else:
        raise ValueError(f'Format {plate_format} is not supported')


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

if __name__ == "__main__":
    main()