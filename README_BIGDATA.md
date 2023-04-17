This repository also contains an example of running the same data set using Apache Spark and Apache SystemML.

In the future Apache SystemML will support using Keras Models to run with the Apache Spark ecosystem.

At the time of this writing there were to many difficulties trying to re-create the models in Keras but in the near future this will be supported.

# Dependencies

This repository is assuming dependencies Java (1.8) and Python 3.6.X and up.

This repository also requires Spark 2.3.X but can use Hadoop 2.6 or 2.7, instructions for installing spark in a UNIX system will be included.

# Installation

```
wget https://archive.apache.org/dist/spark/spark-2.3.3/spark-2.3.3-bin-hadoop2.6.tgz
```
- Curl can be used as an alternative or manual downloading

```
tar -xvzf spark-2.3.3-bin-hadoop2.6.tgz
```

=== Python dependencies ===
We will be relying on SystemML (Version 1.3.0), Sci-kit learn, and PySpark

Sci-kit learn is installed as:

```
pip install scikit-learn
```

for SystemML a specific pip install will be used, as at the time of this writing SystemML is currently on Version 1.2.0 which only supports Spark-2.2. The specific SystemML is put below:

```
pip install https://github.com/niketanpansare/future_of_data/blob/master/systemml-1.3.0-SNAPSHOT-python.tar.gz?raw=true
```

this specific version should only be used until SystemML goes to version 1.3.0.

To install PySpark put:

```
pip install pyspark
```

After pyspark is installed it will need to be pointed to the Spark bin that was downloaded previously. You can do so like:

```
export SPARK_HOME=/path/to/spark-2.3.3-bin-hadoop2.6
```

In a Unix environment, or by adding SPARK_HOME as a Windows Environment Variable.

# Executing

To run the PySpark script you should use spark-submit command:

```
cd code
spark-submit mybigdata_models.py
```

which will display the accuracy per class along with the overall accuracy score

