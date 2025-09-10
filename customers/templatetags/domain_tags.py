from django import template

from customers.models import Domain

register = template.Library()


@register.simple_tag
def domains():
    return Domain.objects.all()
