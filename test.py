import json

def apple_api_full_process(filename_to_read='image_info_test2014.json'):
    """ example of extracting one (small) piece of information from
        the appledata json file...
    """
    f = open( filename_to_read, "r" )
    string_data = f.read()
    data = json.loads( string_data )
    #print("data (not spiffified!) is\n\n", data, "\n")

    # for live investigation, here's the full data structure
    # return data.keys()
    #dict_keys(['licenses', 'categories', 'info', 'images'])

    #return data.get('categories')[0]
    #{'supercategory': 'person', 'name': 'person', 'id': 1}


    #return data.get('info')

    #return data.get('images')[0]
    #{'file_name': 'COCO_test2014_000000523573.jpg', 'width': 423, 'height': 500, 'id': 523573, 'date_captured': '2013-11-14 12:21:59', 'license': 2, 'coco_url': 'http://mscoco.org/images/523573'}



print(apple_api_full_process(filename_to_read='image_info_test2014.json'))

