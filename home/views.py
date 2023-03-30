from django.conf import settings
import stripe
from django.shortcuts import render
from django.http import HttpResponse
import pytesseract
from PIL import Image
from home.models import ChatBot
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import messages
from django.contrib.auth.models import User
import pandas as pd
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
import pandas as pd
from django.contrib.auth import authenticate, login, logout
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from io import BytesIO
import base64
import io
from io import StringIO
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .models import Contact
import PyPDF2
import pytesseract
from PIL import Image
import io
from django.shortcuts import render
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from io import TextIOWrapper, StringIO
import tempfile
import numpy as np
import openai


def faq(request):
    return render(request, 'faq.html')


def blog(request):
    return render(request, 'blog.html')


def home(request):
    return render(request, 'index.html')


def pricing(request):
    return render(request, "listing.html")


def contact(request):
    return render(request, "contact.html")


def services(request):
    return render(request, "services.html")


def payment(request):
    return render(request, "payment.html")


def data_process(request):
    return render(request, "data_process.html")

def plotting(request):
    return render(request, "plotting.html")


def preprocess_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        # Get uploaded CSV file
        csv_file = request.FILES['csv_file']

        # Read CSV file into pandas dataframe
        df = pd.read_csv(TextIOWrapper(
            csv_file.file, encoding='utf-8'), delimiter=',')

        # Perform data augmentation and splitting
        # Replace with your own data augmentation and splitting logic
        augmented_data = df.sample(frac=1).reset_index(drop=True)
        train_df = augmented_data.iloc[:int(len(df)*0.8)]
        test_df = augmented_data.iloc[int(len(df)*0.8):]

        # Save pre-processed CSV files to server
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
            train_df.to_csv(temp, index=False)
            train_file = temp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
            test_df.to_csv(temp, index=False)
            test_file = temp.name

        # Create URLs for pre-processed CSV files
        fs = FileSystemStorage()
        train_url = fs.save(os.path.basename(
            train_file), open(train_file, 'rb'))
        test_url = fs.save(os.path.basename(test_file), open(test_file, 'rb'))

        # Delete temporary files
        os.remove(train_file)
        os.remove(test_file)

        # Return pre-processed CSV file URLs as context variables
        return render(request, 'data_process.html', {'preprocessed_csvs': (fs.url(train_url), fs.url(test_url))})

    # Render empty form
    return render(request, 'data_process.html')


def ocr(request):
    return render(request, "ocr.html")


def visualize(request):
    if request.method == 'POST' and 'csv_file' in request.FILES:
        # Get the uploaded CSV file
        csv_file = request.FILES['csv_file']

        if csv_file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Read XLSX file and convert to CSV
            data = pd.read_excel(csv_file)
            csv_file = io.StringIO()
            data.to_csv(csv_file, index=False)
            csv_file.seek(0)

            # Read the CSV file as a pandas DataFrame
            df = pd.read_csv(csv_file)

            # Save the DataFrame as a CSV file to a BytesIO object
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            # Encode the CSV file as a base64 string for display in the template
            csv_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Render the template with the CSV file data and a form to display the charts
            template = loader.get_template('visualize.html')
            context = {'csv_data': csv_data, 'show_charts': False}
            return HttpResponse(template.render(context, request))

        elif csv_file.content_type == 'application/json':
            # Read JSON file and convert to CSV
            data = pd.read_json(csv_file)
            csv_file = io.StringIO()
            data.to_csv(csv_file, index=False)
            csv_file.seek(0)

            # Read the CSV file as a pandas DataFrame
            df = pd.read_csv(csv_file)

            # Save the DataFrame as a CSV file to a BytesIO object
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            # Encode the CSV file as a base64 string for display in the template
            csv_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Render the template with the CSV file data and a form to display the charts
            template = loader.get_template('visualize.html')
            context = {'csv_data': csv_data, 'show_charts': False}
            return HttpResponse(template.render(context, request))

        else:

            # Read the CSV file as a pandas DataFrame
            df = pd.read_csv(csv_file)

            # Save the DataFrame as a CSV file to a BytesIO object
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            # Encode the CSV file as a base64 string for display in the template
            csv_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Render the template with the CSV file data and a form to display the charts
            template = loader.get_template('visualize.html')
            context = {'csv_data': csv_data, 'show_charts': False}
            return HttpResponse(template.render(context, request))

    elif request.method == 'POST' and 'display_charts' in request.POST:
        # Get the CSV file data from the form
        csv_data = request.POST['csv_data']

        # Decode the base64-encoded CSV file data
        buffer = BytesIO(base64.b64decode(csv_data.encode('utf-8')))

        # Read the CSV file as a pandas DataFrame
        df = pd.read_csv(buffer)

        # Generate the requested chart and save it to a BytesIO object
        chart_type = request.POST['chart_type']
        if chart_type == 'box_plot':
            fig, ax = plt.subplots()
            df.boxplot(ax=ax)
        elif chart_type == 'bubble_chart':
            # Define the columns for the bubble chart
            x_col = df.columns[0]  # First column
            y_col = df.columns[1]  # Second column
            size_col = df.columns[2]  # Third column
            color_col = df.columns[3]  # Fourth column

            # Create the bubble chart
            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col], s=df[size_col]
                       * 100, c=df[color_col], alpha=0.5)

            # Set the axis labels and title
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Bubble Chart")

        elif chart_type == 'time_series_chart':
            time_col = pd.to_datetime(df.columns[0]) if pd.api.types.is_datetime64_ns_dtype(
                df.columns[0]) else pd.to_datetime(df.iloc[:, 0])
            ts_data = pd.Series(df.iloc[:, 1].values, index=time_col)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ts_data)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title('Time Series Plot')
            canvas = FigureCanvas(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            response = HttpResponse(buf.getvalue(), content_type='image/png')
            response['Content-Length'] = len(response.content)
            return response
        elif chart_type == 'graph_chart':
            G = nx.DiGraph()
            for i, row in df.iterrows():
                G.add_edge(row[0], row[1])
            fig, ax = plt.subplots()
            pos = nx.spring_layout(G, k=0.15, seed=100)
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=500, node_color='lightblue', alpha=0.9)
            nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=10, arrowstyle='fancy', width=df.get(
                'edge_width', 1), edge_color=df.get('edge_color', 'k'))
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
            # save the plot to a bytes object
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
        elif chart_type == 'correlation_matrix':
            corr_matrix = df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, ax=ax)
        elif chart_type == 'bar_chart':
            x_col = df.columns[0]
            y_col = df.columns[1]
            fig, ax = plt.subplots()
            ax.bar(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Bar Chart")
            plt.xticks(rotation=90)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
        elif chart_type == 'scatter_plot':
            x_col = df.columns[0]
            y_col = df.columns[1]
            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Scatter Plot")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

        elif chart_type == 'line_graph':
            x_col = df.columns[0]
            y_col = df.columns[1]
            fig, ax = plt.subplots()
            ax.plot(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Line Graph")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
        elif chart_type == 'pie_chart':
            x_col = df.columns[0]
            y_col = df.columns[1]
            fig, ax = plt.subplots()
            ax.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
            ax.set_title("Pie Chart")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
        elif chart_type == 'heat_map':
            x_col = df.columns[0]
            y_col = df.columns[1]
            fig, ax = plt.subplots()
            sns.heatmap(df, ax=ax)
            ax.set_title("Heat Map")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
        elif chart_type == 'area_chart':
            x_col = df.columns[0]
            y_col = df.columns[1]
            fig, ax = plt.subplots()
            ax.fill_between(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Area Chart")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
        elif chart_type == 'choropleth_map':
            country_data = df.groupby('Country').sum()
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.axis('off')
            ax.set_title('Choropleth Map', fontdict={
                'fontsize': '25', 'fontweight': '3'})
            data = country_data['value']
            cmap = plt.cm.Blues
            scheme = [cmap(i / len(data)) for i in range(len(data))]
            countries = list(country_data.index)
            ax.pie(data, labels=countries, colors=scheme)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
        elif chart_type == 'histogram':
            x_col = df.columns[0]
            fig, ax = plt.subplots()
            ax.hist(df[x_col])
            ax.set_xlabel(x_col)
            ax.set_title("Histogram")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

        else:
            # Default to a bar chart if the chart type is not recognized
            fig, ax = plt.subplots()
            ax.bar(df['x'], df['y'])

        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the chart image as a base64 string for display in the template
        chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Render the template with the CSV file data and the chart image
        template = loader.get_template('visualize.html')

        # Update the context to display the chart image
        context = {'csv_data': csv_data,
                   'chart_data': chart_data, 'show_charts': True}
        return HttpResponse(template.render(context, request))

    else:
        # Render the template with a form to upload a CSV file
        template = loader.get_template('visualize.html')
        context = {'show_charts': False}
        return HttpResponse(template.render(context, request))


def visualize_page(request):
    return render(request, "visualize.html")


def terms_condition(request):
    return render(request, "terms_condition.html")


def confusion_matrix(request):
    return render(request, "confusion_matrix.html")


def roc_auc(request):
    return render(request, "roc_auc.html")


def dataset(request):
    return render(request, "dataset.html")


def error_analysis(request):
    return render(request, "error_analysis.html")


def dataset_description(request):
    return render(request, "dataset_description.html")


def data_cleaning(request):
    return render(request, "data_cleaning.html")


def handle_null_values(df):
    # Get list of columns with null values
    null_columns = df.columns[df.isnull().any()]

    # Replace null values with mean for numerical columns
    for col in null_columns:
        if df[col].dtype == np.float64 or df[col].dtype == np.int64:
            mean = df[col].mean()
            df[col] = df[col].fillna(mean)

    # Replace null values with mode for categorical columns
    for col in null_columns:
        if df[col].dtype == object:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)

    return df


def convert_to_float(df):
    # Loop through dataframe columns
    for col in df.columns:
        # Check if column contains string values
        if df[col].dtype == object:
            # Convert string values to float
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

    return df


def cleaning(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        # Get uploaded CSV file
        csv_file = request.FILES['csv_file']

        # Read CSV file into pandas dataframe
        df = pd.read_csv(TextIOWrapper(
            csv_file.file, encoding='utf-8'), delimiter=',')

        # Handle null values
        df = handle_null_values(df)

        # Convert string columns to float
        df = convert_to_float(df)

        # Save pre-processed CSV file to server
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir='media') as temp:
            df.to_csv(temp, index=False)
            csv_url = FileSystemStorage().url(temp.name)

        # Return pre-processed CSV file URL as context variable
        return render(request, 'data_cleaning.html', {'preprocessed_csv': csv_url})

    # Render empty form
    return render(request, 'data_cleaning.html')




def chat(request):
    msg = ChatBot.objects.filter(username=request.user)
    return render(request, "chatbot.html", {"msg": msg})


def chat_form(request):
    response = None
    if request.method == 'POST':
        message = request.POST.get('message')
        openai.api_key = "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"
        prompt = (f"User: {message}\n"
                  "Chatbot:")
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = completions.choices[0].text
        response = message.replace("User: ", "")
    return render(request, 'robot.html', {'response': response})





# SService
def fully_responsive(request):
    return render(request, "fully_responsive.html")


def fresh_layout(request):
    return render(request, "fresh_layout.html")


def minimalism_feast(request):
    return render(request, "minimalism_feast.html")


def modern_workflow(request):
    return render(request, "modern_workflow.html")


def unique_feature(request):
    return render(request, "unique_feature.html")


def support(request):
    return render(request, "support.html")


def team(request):
    return render(request, "team.html")

# Contact


def contact_form(request):
    if request.method == "POST":
        name = request.POST['name']
        phone = request.POST['phone']
        email = request.POST['email']
        company = request.POST['company']
        msg = request.POST['message']

        contact_database = Contact(
            name=name, phone=phone, email=email, company=company, msg=msg)
        contact_database.save()
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


def signinn(request):
    if request.method == 'POST':
        name = request.POST['name']
        password = request.POST['password']
        user = authenticate(request, username=name, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')
        else:
            messages.error(request, 'Email or Password incorrect')

    return render(request, 'login.html')


def signup(request):
    if request.method == "POST":
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already taken")
            elif User.objects.filter(email=email).exists():
                messages.error(request, "Email already taken")
            else:
                user = User.objects.create_user(
                    first_name=first_name, last_name=last_name, username=username, password=password, email=email)
                user.save()
                login(request, user)

                return redirect('/')

        else:
            messages.error(request, 'Password not matched')

    return render(request, 'signup.html')


def signout(request):
    logout(request)
    return redirect("/")


def forgot(request):
    forgot(request)
    return redirect("/")


def OCR(request):
    # Check if image was uploaded
    if request.method == 'POST' and request.FILES['image']:
        # Read the image file from the request
        image = Image.open(request.FILES['image'])

        # Convert the image to grayscale
        image = image.convert('L')

        # Perform OCR on the image
        text = pytesseract.image_to_string(image)

        # Render the OCR result in the response
        return render(request, 'ocr.html', {'text': text})

    # If image was not uploaded, render the upload form
    return render(request, 'upload.html')


from django.shortcuts import render
import stripe

stripe.api_key = settings.STRIPE_SECRET_KEY

def charge(request):
    if request.method == 'POST':
        package = request.POST['package']
        amount = 0
        if package == 'basic':
            amount = 1000
        elif package == 'standard':
            amount = 2000
        elif package == 'premium':
            amount = 3000
        else:
            # Handle invalid package
            pass
        try:
            charge = stripe.Charge.create(
                amount=amount,
                currency='usd',
                description='Package Purchase',
                source=request.POST['stripeToken']
            )
            # Save package information to your database
            # ...
            return render(request, 'success.html')
        except stripe.error.CardError as e:
            # Handle card error
            pass
    return render(request, 'checkout.html')


