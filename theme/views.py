from django.shortcuts import render


def home(request):
    """Render the project homepage."""

    return render(request, "theme/home.html")
