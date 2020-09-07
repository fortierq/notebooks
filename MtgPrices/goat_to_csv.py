# written to be used with Python 3
from pprint import pprint

import csv
import json
import pathlib
import random
import time

import requests


def main():
    output_dir = 'cards'
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    
    # this list was semi-manually created from the URLS at https://www.goatbots.com/prices
    card_names = [
        'watery-grave'
    ]

    for name_of_card in card_names:
        json_url = f'https://www.goatbots.com/card/ajax_card?search_name={name_of_card}'
        print(f'getting: {name_of_card}')

        time.sleep(random.randint(2, 6))
        response = requests.get(json_url)

        try:
            content = json.loads(response.content)
        except json.decoder.JSONDecodeError:
            print(f'yeah, that failed for: {name_of_card}')
            continue

#        pprint(content[1][0])

        with open(f'{output_dir}/{name_of_card}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            rows = []

            for row in content[1][0]:
                day_1 = row[0]
                price_1 = row[1]

                if len(content[1]) == 2:
                    for line in content[1][1]:
                        day_2 = line[0]
                        price_2 = line[1]
                        if day_1 == day_2:
                            rows.append([day_1, price_1, price_2])
                            # print(f'breaking because {day_1} == {day_2}')
                            break
                    else:
                        rows.append([day_1, price_1])

                elif len(content[1]) == 2:
                    rows.append([day_1, price_1])

            # second pass, to make sure we get everything!
            if len(content[1]) == 2:
                for line in content[1][1]:
                    day_2 = line[0]
                    price_2 = line[1]
                    for row in rows:
                        day_1 = row[0]
                        if day_1 == day_2:
                            break
                    else:
                        rows.append([day_2, price_2])

            for row in rows:
                csvwriter.writerow(row)


if __name__ == '__main__':
    main()
