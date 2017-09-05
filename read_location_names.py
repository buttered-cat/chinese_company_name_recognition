import re


def read_location_names():
    locations = []
    locations_with_suffix = []
    location_suffix_pattern = r'省'
    # location_suffix_pattern = r'(省)|(市)|(区)|(县)|(自治县)|(自治州)'
    with open('./data/LocationNames.txt', 'r', encoding='utf8') as data:
        for line in data.readlines():
            if re.match(r'\d+\.', line) is not None:
                province = line.split('.')[1].split()[0]
                locations_with_suffix.append(province)
                province = re.sub(location_suffix_pattern, '', province)
                locations.append(province)
            elif re.match(r'\d{4}.+。', line) is not None:
                continue
            else:
                locations_inline = line.split()
                locations_with_suffix += locations_inline
                for location in locations_inline:
                    location = re.sub(location_suffix_pattern, '', location)
                    locations.append(location)

    # with open('./data/location_name_data.txt', 'w', encoding='utf8') as f:
    #     for location in locations:
    # returns unsegmented strings
    return locations, locations_with_suffix

read_location_names()
# print(len(locations))
