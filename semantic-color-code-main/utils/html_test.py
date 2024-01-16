from jinja2 import Environment, FileSystemLoader 
 
def generate_html(data):
    env = Environment(loader=FileSystemLoader('./'))
    template = env.get_template('/home/amax/Documents/semantic-color-code/utils/index.html')     
    with open("./result.html",'w+') as fout:   
        html_content = template.render(data=data)
        fout.write(html_content)
 
if __name__ == "__main__":
    data = ""   
    generate_html(data) 