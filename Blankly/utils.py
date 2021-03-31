"""
    Utils file for assisting with trades or market analysis.
    Copyright (C) 2021  Emerson Dove

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import datetime as DT
import json
import numpy
import time

import iso8601
from sklearn.linear_model import LinearRegression


# def printJSON(jsonObject):
#     """
#     Json pretty printer for show arguments
#     """
#     print(pretty_print_JSON(jsonObject))


def pretty_print_JSON(json_object):
    """
    Json pretty printer for general string usage
    """
    out = json.dumps(json_object, indent=2)
    print(out)
    return out


def epoch_from_ISO8601(ISO8601):
    return time.mktime(iso8601.parse_date(ISO8601).timetuple())


def ISO8601_from_epoch(epoch):
    return DT.datetime.utcfromtimestamp(epoch).isoformat() + 'Z'


def getPriceDerivative(ticker, point_number):
    """
    Performs regression n points back
    """
    feed = numpy.array(ticker.get_ticker_feed()).reshape(-1, 1)
    times = numpy.array(ticker.get_time_feed()).reshape(-1, 1)
    if point_number > len(feed):
        point_number = len(feed)

    feed = feed[-point_number:]
    times = times[-point_number:]
    prices = []
    for i in range(point_number):
        prices.append(feed[i][0]["price"])
    prices = numpy.array(prices).reshape(-1, 1)

    regressor = LinearRegression()
    regressor.fit(times, prices)
    regressor.predict(times)
    return regressor.coef_[0][0]


def fitParabola(ticker, point_number):
    """
    Fit simple parabola
    """
    feed = ticker.get_ticker_feed()
    times = ticker.get_time_feed()
    if point_number > len(feed):
        point_number = len(feed)

    feed = feed[-point_number:]
    times = times[-point_number:]
    prices = []
    for i in range(point_number):
        prices.append(float(feed[i]["price"]))
        times[i] = float(times[i])

    # Pull the times back to x=0 so we can know what happens next
    latest_time = times[-1]
    for i in range(len(prices)):
        times[i] = times[i] - latest_time

    return numpy.polyfit(times, prices, 2, full=True)