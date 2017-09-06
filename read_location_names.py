import re


def read_location_names():
    locations_without_suffix = []
    locations_with_suffix = []
    location_suffix_pattern = r'省'
    # location_suffix_pattern = r'(省)|(市)|(区)|(县)|(自治县)|(自治州)'
    with open('./data/LocationNames.txt', 'r', encoding='utf8') as data:
        for line in data.readlines():
            if re.match(r'\d+\.', line) is not None:
                province = line.split('.')[1].split()[0]
                locations_with_suffix.append(province)
                province_without_suffix = re.sub(location_suffix_pattern, '', province)
                if province_without_suffix != province:
                    locations_without_suffix.append(province_without_suffix)
            elif re.match(r'\d{4}.+。', line) is not None:
                continue
            else:
                locations_inline = line.split()
                locations_with_suffix += locations_inline
                for location in locations_inline:
                    location_without_suffix = re.sub(location_suffix_pattern, '', location)
                    if location_without_suffix != location:
                        locations_without_suffix.append(location_without_suffix)

    # with open('./data/location_name_data.txt', 'w', encoding='utf8') as f:
    #     for location in locations:
    # returns unsegmented strings

    # mutually exclusive
    return locations_with_suffix, locations_without_suffix

# w, wt = read_location_names()
# pass
# print(len(locations))
