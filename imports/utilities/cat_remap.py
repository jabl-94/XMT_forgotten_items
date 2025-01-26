def remap_categories(data, id_to_name_map):
    remapped_data = {}
    for customer_id, customer_data in data.items():
        remapped_customer_data = {}
        for basket_id, basket_data in customer_data['data'].items():
            remapped_basket = {id_to_name_map.get(item_id, item_id): count
                               for item_id, count in basket_data['basket'].items()}
            remapped_customer_data[basket_id] = {'basket': remapped_basket}
        remapped_data[customer_id] = {'data': remapped_customer_data}
    return remapped_data