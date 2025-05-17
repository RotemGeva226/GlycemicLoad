from chains.langchain_flow import final_chain

if __name__ == "__main__":
    image_path = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\PortionsEstimation\src\data\raw\single_ingredient_images\dish_1558029686.jpg"

    result = final_chain.invoke(image_path)
    print(result)