from django.shortcuts import render


def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")
