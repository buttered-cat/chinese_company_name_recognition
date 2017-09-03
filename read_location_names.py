import re

# TODO: separate location names from suffixes & correctly recognize


def read_location_names():
    locations = []
    with open('./data/LocationNames.txt', 'r', encoding='utf8') as data:
        for line in data.readlines():
            if re.match(r'\d+\.', line) is not None:
                province = locations.append(line.split('.')[1].split()[0])
            elif re.match(r'\d{4}.+。', line) is not None:
                continue
            else:
                locations += line.split()
                # for location in locations:
                #     p = re.compile(r'(区)|(县)|(自治县)|(自治州)')

    # with open('./data/location_name_data.txt', 'w', encoding='utf8') as f:
    #     for location in locations:
    return locations

# print(len(locations))
