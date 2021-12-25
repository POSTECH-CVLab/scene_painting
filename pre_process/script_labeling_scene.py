"""
1. Parse ade20k label file (/home/jbjeong/projects/scene3d/sceneparsing/objectInfo150.csv').
2. Assign a label to objects in the scene.
"""
import os

import bpy


# 1Living-room/living_room_11.obj
living_room_special_dict = {
    'TV': 90, # television
    'ceiling_bulbs': 83, # light
    'ceiling_lights': 83, # light
    'cooker': 119, # oven
    'curtains': 19, # curtain
    'deco': 133, # sculpture
    'deco.001': 126, # flowerpot
    'drawer': 45, # drawers
    'drawer.001': 45, # drawers
    'exhaust_chimney': 50, # fireplace
    'faucet': 48, # sink
    'faucet.001': 48, # sink
    'fridge': 51, # refrigerator
    'fruit_bowl': 130, # dish  
    'furniture': 11, # cabinet 
    'gas_stove': 72, # stove
    'glasses': 148, # glass
    'kettle': 126, # pot
    'kitchen_furniture': 100, # sideboard
    'kitchen_furniture.000': 100, # sideboard
    'kitchen_lights': 83, # light
    'kitchen_shelf': 25, # shelf 
    'kitchen_shelf.000': 25, # shelf 
    'knife': 72, # kitchen 
    'notebook': 68, # book
    'paintings': 23, # painting
    'pen': 68, # book
    'pillows': 58, # pillow
    'plate_rack': 143, # plate
    'plates': 143, # plate
    'speakers': 90, # television
    'window_blinds': 87, # sunblind
    'windows': 9, # window
    'windows.001': 9, # window
    'wine_bottle': 99, # bottle
    'wine_bottle.000': 99, # bottle
    'wine_bottles': 99, # bottle
    'wine_rack': 99, # bottle
}

# 1Office/66office_scene.obj
office_special_dict = {
    'Bin': 139, # wastebin
    'Bin.001': 139, # wastebin
    'Bin.002': 139, # wastebin
    'Bin.003': 139, # wastebin
    'File_rack': 25, # shelf 
    'File_rack.001': 25, # shelf 
    'File_rack.002': 25, # shelf 
    'File_rack.003': 25, # shelf 
    'File_rack.004': 25, # shelf 
    'File_rack.005': 25, # shelf 
    'Files': 68, # book
    'Files.001': 68, # book
    'Files.002': 68, # book
    'Files.003': 68, # book
    'Files.004': 68, # book
    'Files.005': 68, # book
    'Files.006': 68, # book
    'Files.007': 68, # book
    'Files.008': 68, # book
    'Files.009': 68, # book
    'Files.010': 68, # book
    'Files.011': 68, # book
    'Files.012': 68, # book
    'Files.013': 68, # book
    'Floor': 4, # floor
    'blinds.011': 87, # sunblind
    'blinds.012': 87, # sunblind
    'blinds.013': 87, # sunblind
    'books.074': 68, # book 
    'books.075': 68, # book 
    'books.076': 68, # book 
    'books.077': 68, # book 
    'books.078': 68, # book 
    'books.079': 68, # book 
    'books.080': 68, # book 
    'books.081': 68, # book 
    'books.082': 68, # book 
    'books.083': 68, # book 
    'books.084': 68, # book 
    'ceiling_lights': 83, # light 
    'ceiling_lights.001': 83, # light 
    'chair1': 20, # chair
    'chair2': 20, # chair
    'chair3': 20, # chair
    'chair4': 20, # chair
    'chair5': 20, # chair
    'chair6': 20, # chair
    'chair7': 20, # chair
    'chair8': 20, # chair
    'chair9': 20, # chair
    'chair10': 20, # chair
    'computer_cabinet': 11, # cabinet
    'computer_cabinet.001': 11, # cabinet
    'computer_cabinet.002': 11, # cabinet
    'decoration.001': 136, # vase
    'drawer': 45, # drawers
    'drawer.001': 45, # drawers
    'drawer.002': 45, # drawers
    'drawer.003': 45, # drawers
    'drawer.004': 45, # drawers
    'drawer.005': 45, # drawers
    'drawer.006': 45, # drawers
    'drawer.007': 45, # drawers
    'notebook': 68, # book 
    'notebook.001': 68, # 
    'notebook.002': 68, # 
    'notebook.003': 68, # 
    'notebook.004': 68, # 
    'notebook.005': 68, # 
    'notebook.006': 68, #
    'notebook.007': 68, # 
    'notebook.008': 68, # book
    'obj': 133, # sculpture
    'obj.000': 133, # sculpture
    'obj.001': 133, # sculpture
    'obj.002': 133, # sculpture
    'pen': 34, # desk
    'pen.001': 34, # desk
    'pen.007': 34, # desk
    'pen.008': 34, # desk
    'pen.009': 34, # desk
    'pen.010': 34, # desk
    'pen.011': 34, # desk
    'pen.012': 34, # desk
    'pen.013': 34, # desk
    'pen.014': 34, # desk
    'printer.001': 75, # computer
    'printer.003': 75, # computer
    'printer.004': 75, # computer
    'printer.005': 75, # computer
    'windows': 9, # window
    'windows.001': 9, # window
    'windows.002': 9, # window
    'windows.003': 9, # window
}

# 1Bedroom/bedroom_wenfagx.obj
bedroom_special_dict = {
    'Teapot01.001': 126, # pot
    'Teapot02.001': 126, # pot
    'brushes.001': 67, # flower
    'candle.001': 135, # sconce
    'candle_holder.001': 136, # vase
    'chest_of_drawer': 45, # drawer
    'cup.001': 148, # glass
    'cup.002': 148, # glass
    'cup.003': 148, # glass
    'cupboard.003': 36, # closet
    'cupboard.004': 36, # closet
    'curtain_rod.001': 94, # pole
    'deco1.001': 126, # pot
    'duvet.003': 132, # blanket
    'duvet.004': 132, # blanket
    'furniture.001': 100, # sideboard
    'hanger.016': 93, # apparel
    'hanger.017': 93, # apparel
    'hanger.018': 93, # apparel
    'hanger.019': 93, # apparel
    'hanger.020': 93, # apparel
    'hanger.021': 93, # apparel
    'hanger.022': 93, # apparel
    'hanger.023': 93, # apparel
    'hanger.024': 93, # apparel
    'hanger.025': 93, # apparel
    'hanger.026': 93, # apparel
    'hanger.027': 93, # apparel
    'hanger.028': 93, # apparel
    'hanger.029': 93, # apparel
    'hanger.030': 93, # apparel
    'hanger.031': 93, # apparel
    'hanging_rod.001': 94, # pole
    'tea_cup_and_plate.001': 148 # glass
}


# 1Bathroom/28_labels.obj
bathroom_special_dict = {
    'cupboard': 36, # closet
    'decoration': 133, # sculpture
    'decoration.001': 136, # vase
    'decoration.002': 136, # vase
    'decoration.003': 136, # vase
    'decoration.004': 136, # vase
    'shower_cabin': 89, # booth
    'shower_light_fan': 135, # sconce
    'soap': 109, # toy
    'soap_dish': 143, # plate 
    'tap': 48, # sink
    'tap.001': 48, # sink
    'toothbrush': 109, # toy
    'toothbrush_holder': 136, # vase,
    'track_light': 83, # light
    'track_light.001': 83, # light
    'track_light.002': 83, # light
    'track_light.003': 83, # light
}

# 1Kitchen/kitchen_76_blender_name_and_mat.obj
kitchen_special_dict = {
    'Gas1': 72, # stove 
    'Heater1': 147, # radiator
    'Shelf1': 71, # countertop
    'SupportWall': 25, # shelf
    'Tap1': 48, # sink
    'Tray': 71, # countertop
    'UtensilHangers': 94, # pole
    'WineStand': 74, # kitchen island
    'Container1': 123, # tank
    'Cupboard': 100, # sideboard 
    'Jar1': 148, # glass
    'Jar2': 126, # pot  
    'Jar3': 126, # pot 
    'Kettle': 126, # pot
    'Kettle.001': 126, # pot
    'KitchenUtensil1': 74, # kitchen island
    'Pan1': 74, # kitchen island
    'Pan2': 74, # kitchen island
    'PressureCooker': 123, # tank
    'Spatula1': 74, # kitchen island 
    'Spatula2': 74, # kitchen island
}

['Container1', 'Cupboard', 'Jar1', 'Jar2', 'Jar3', 'Kettle1', 'Kettle3', 'KitchenUtensil1', 'Pan1', 'Pan2', 'PressureCooker', 'Spatula1', 'Spatula2']

if __name__=='__main__':

    # Parse ade20k label file.
    ade20k_label_dict = {} # key: label, value: obj_name_list
    ade20k_total_name_list = []
    with open('/home/jbjeong/projects/scene3d/sceneparsing/objectInfo150.csv') as fp:
        while True:
            line = fp.readline()
            if not line: break

            line = line.strip()
            split_line = line.split(',')
            label = split_line[0]
            if label == 'Idx': continue # Skip the first row 
            label = int(label)
            name_list = split_line[-1].split(';')

            ade20k_label_dict[label] = name_list
            ade20k_total_name_list += name_list

    scene_file_name = os.path.basename(bpy.data.filepath)

    if scene_file_name == '1Living-room_living_room_11.blend':
        special_dict = living_room_special_dict
    elif scene_file_name == '1Office_66office_scene.blend':
        special_dict = office_special_dict 
    elif scene_file_name == '1Bedroom_bedroom_wenfagx.blend':
        special_dict = bedroom_special_dict 
    elif scene_file_name == '1Bathroom_28_labels.blend':
        special_dict = bathroom_special_dict 
    elif scene_file_name == '1Kitchen_kitchen_76_blender_name_and_mat.blend':
        special_dict = kitchen_special_dict 

    scene = bpy.context.scene
    all_objects = scene.collection.all_objects

    is_debug = False 
    if is_debug:

        special_name_list = []
        for key, value in all_objects.items():
            if key == 'Camera': continue

            if not key.split('.')[0] in ade20k_total_name_list:
                special_name_list.append(key)

        special_name_list = sorted(special_name_list)

        special_dict_names = list(special_dict.keys())
        remove_name_list = []
        for name in special_name_list:
            #if name in special_dict_names or name.split('.')[0] in special_dict_names:
            if name in special_dict_names:
                remove_name_list.append(name)

        for name in remove_name_list:
            special_name_list.remove(name)

        print(len(special_name_list))
        print(special_name_list)
        import pdb; pdb.set_trace()

    for key, value in all_objects.items():
        if key == 'Camera': continue

        for label, name_list in ade20k_label_dict.items():
            if key.split('.')[0] in name_list:
                value.pass_index = label
                break

        if value.pass_index == 0:
            for s_name, label in special_dict.items():
                if key == s_name:
                    value.pass_index = label
                    break

        assert value.pass_index != 0, f'Error in pass_index of key {key}'

    print('Finish')
    import pdb; pdb.set_trace()

