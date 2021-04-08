---
title: Mybatis
date: 2019-12-26 10:43:57
tags: java
---

## 框架

1. 框架的概念:软件开发中的一套解决方案，不同方案可以解决不同的问题，封装了许多的细节，可以使开发效率大幅度提高。
2. 三层架构：
   - 表现层：数据的展示
   - 业务层：处理业务需求
   - 持久层：与数据库交互
3. 持久层的技术解决方案
   - JDBC技术
     - Connection
     - PreparedStatement
     - ResultSet
   - Spring的JdbcTemplate:Spring对jdbc的简单封装
   - Apache的DBUtiles:与SPring的jdbcTemplate很像，也是对与jdbc的简单封装
4. mybatis是一个持久层的框架，封装了jdbc操作的细节，使用了ORM思想，实现了结果集的封装。
  ORM(Object Relational Mapping)将数据库表和实体类及实体类的属性对应起来可以操作实体类就实现操作数据库表

## 环境搭建

1. 使用IDE创建出一个maven项目
2. 选择项目地打包方式，在项目中引入依赖

 ``` java
   <groupId>com.itheima</groupId>
    <artifactId>first_mybatis</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>
    <dependencies>
        <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis</artifactId>
        <version>3.4.5</version>
    </dependency>
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>6.0.5</version>
        </dependency>
        <dependency>
            <groupId>log4j</groupId>
            <artifactId>log4j</artifactId>
            <version>1.2.12</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.12</version>
        </dependency>
    </dependencies>
```

1. 接下来在src中创建数据库表中对应的对象，且要implements Serializable使对象序列化接下来是一些基本你的操作，将所有属性私有化留出get和set方法

2. 接下来创建出一个IuserDao的接口(I的意思只是Interface)称为用户的持久层接口，在其中创建查询所有操作的序列。

3. 配置mybtis的环境，创建一个xml文件在resources文件夹下,在创建完成后需要在相应的mapper位置下建立对应的配置，且引入mapper的约束。

Config配置

``` xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration
        PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
    <!--配置环境-->
    <environments default="mysql">
        <!--配置mysql环境-->
        <environment id="mysql">
            <!--配置事务类型-->
            <transactionManager type="JDBC"></transactionManager>
            <!--配置连接池-->
            <dataSource type="POOLED">
                <!--配置连接数据库的基本信息-->
                <property name="driver" value="com.mysql,jdbc.Driver"/>
                <property name="url" value="jdbc:mysql://localhost:3306/mybatis"/>
                <property name="username" value="root"/>
                <property name="password" value="root"/>
            </dataSource>
        </environment>
    </environments>

    <!--指定映射配置文件-->
    <mappers>
        <mapper resource="com/itheima.dao/IuserDao.xml"/>
    </mappers>
</configuration>
```

mapper的配置

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper
        PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-mapper.dtd" >

    <mapper namespace="com.itheima.dao.IUserDao">
    <!--配置查询所有-->
    <select id="findAll">
        select * from USER
    </select>
</mapper>
```

至此环境搭建基本完成，在进行这些操作前需要将maven项目的环境搭建好，并建立好仓库等。

## mybatis的实例

1.在test的java文件夹中以main函数来进行测试，且使用了工厂模式来进行sql语句的执行

``` java
public static void main(String[] args) throws Exception{
        //1.读取配置文件
        InputStream inputStream = Resources.getResourceAsStream("SqlMapConfig.xml");
        //2.创建SQL Session Factory工厂
        SqlSessionFactoryBuilder builder = new SqlSessionFactoryBuilder();
        SqlSessionFactory factory = builder.build(inputStream);
        //3.使用工厂生产SqlSession对象
        SqlSession session = factory.openSession();
        //4.使用SQL Session创建Dao接口代理对象
        IUserDao userDao = session.getMapper(IUserDao.class);
        //5.使用代理对象执行方法
        List<User> users;
        users = userDao.findAll();
        for (User user : users){
            System.out.println(user);
        }
        //6.释放资源
        session.close();
        inputStream.close();
    }
  ```

执行后发现报错并不如愿，理想中的报错应该是返回由于映射的没有返回结果而报错，最终发现是自己的mapper地址写错，以及jdbc的驱动加载写错。

- mybatis的映射配置文件位置必须和dao接口的包接口相同

- 映射配置文件mapper的标签namespace属性取值必须是dao接口的全限定类名

- 映射配置文件的操作配置，id属性取值必须是dao接口的方法名。

### 实例中使用到的几个类

1. class Resources
2. class SqlSessionFactoryBuilder
3. interface SqlSessionFactory
4. interface SqlSession

在符合上面的几个条件的情况下，就不需要再写dao的实现类了。

### 注解方式来使用mybatis

则可以删除每个dao对应的dao.xml改为@select注解，并且指定SQL语句，在config文件中配置mappper时，使用class属性指定dao接口的全限定类名。

## 自定义MyBatis

1. 首先需要有数据库的连接信息，通过连接信息，可以创建Connection对象，接下来需要映射的配置信息，最后从映射中配置需要的SQL语句。就可以获取PreparedStatement。通过dom4j来解析这些xml配置。

2. 根据配置文件的细腻些创建Connection对象，获取预处理对象PrepareSatement接下来执行查询，将遍历的结果集封装，将结果返回。
