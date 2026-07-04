-- Automatically drop the old conflicting database if it exists
DROP DATABASE IF EXISTS `sinhala_tts_db`;

-- Create fresh Database
CREATE DATABASE IF NOT EXISTS `sinhala_tts_db` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE `sinhala_tts_db`;

-- 1. USER Table 
CREATE TABLE IF NOT EXISTS `USER` (
    `User_ID` INT AUTO_INCREMENT PRIMARY KEY,
    `Role` VARCHAR(20) NOT NULL
) ENGINE=InnoDB;

-- 2. REGISTERED_USER Table
CREATE TABLE IF NOT EXISTS `REGISTERED_USER` (
    `User_ID` INT PRIMARY KEY,
    `Email` VARCHAR(100) NOT NULL UNIQUE,
    `User_Name` VARCHAR(50) NOT NULL UNIQUE,
    `Password_Hash` VARCHAR(255) NOT NULL,
    `Account_Status` VARCHAR(20) NOT NULL DEFAULT 'Active',
    `Failed_Login_Count` INT DEFAULT 0,
    `Lock_Timestamp` DATETIME NULL,
    FOREIGN KEY (`User_ID`) REFERENCES `USER`(`User_ID`)
) ENGINE=InnoDB;

-- 3. GUEST Table
CREATE TABLE IF NOT EXISTS `GUEST` (
    `User_ID` INT PRIMARY KEY,
    `Session_ID` VARCHAR(100) NOT NULL UNIQUE,
    `IP_Address` VARCHAR(45) NOT NULL,
    `Created_At` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (`User_ID`) REFERENCES `USER`(`User_ID`)
) ENGINE=InnoDB;

-- 4. QUERY Table
CREATE TABLE IF NOT EXISTS `QUERY` (
    `Query_ID` INT AUTO_INCREMENT PRIMARY KEY,
    `Input_Text` TEXT NOT NULL,
    `Syllabus_Topic` VARCHAR(100) NOT NULL,
    `Timestamp` DATETIME NOT NULL,
    `User_ID` INT NULL,
    FOREIGN KEY (`User_ID`) REFERENCES `USER`(`User_ID`)
) ENGINE=InnoDB;

-- 5. ANSWER Table
CREATE TABLE IF NOT EXISTS `ANSWER` (
    `Answer_ID` INT AUTO_INCREMENT PRIMARY KEY,
    `Response_Text` TEXT NOT NULL,
    `Source` VARCHAR(100) NULL,
    `Query_ID` INT UNIQUE,
    FOREIGN KEY (`Query_ID`) REFERENCES `QUERY`(`Query_ID`)
) ENGINE=InnoDB;

-- 6. AUDIO Table
CREATE TABLE IF NOT EXISTS `AUDIO` (
    `Audio_ID` INT AUTO_INCREMENT PRIMARY KEY,
    `File_Path` VARCHAR(255) NOT NULL,
    `Format` VARCHAR(10) NULL,
    `Duration` INT NULL,
    `Model_Version` VARCHAR(50) NULL,
    `Processing_Time` DECIMAL(6,2) NULL,
    `Answer_ID` INT UNIQUE,
    FOREIGN KEY (`Answer_ID`) REFERENCES `ANSWER`(`Answer_ID`)
) ENGINE=InnoDB;

-- 7. DOWNLOAD_LOG Table
CREATE TABLE IF NOT EXISTS `DOWNLOAD_LOG` (
    `Log_ID` INT AUTO_INCREMENT PRIMARY KEY,
    `Date` DATETIME NOT NULL,
    `Audio_ID` INT NULL,
    `User_ID` INT NULL,
    FOREIGN KEY (`Audio_ID`) REFERENCES `AUDIO`(`Audio_ID`),
    FOREIGN KEY (`User_ID`) REFERENCES `USER`(`User_ID`)
) ENGINE=InnoDB;

-- 8. USER_FEEDBACK Table
CREATE TABLE IF NOT EXISTS `USER_FEEDBACK` (
    `Feedback_ID` INT AUTO_INCREMENT PRIMARY KEY,
    `Rating` INT NOT NULL,
    `Comment` TEXT NULL,
    `Timestamp` DATETIME NOT NULL,
    `Audio_ID` INT NULL,
    `User_ID` INT NULL,
    FOREIGN KEY (`Audio_ID`) REFERENCES `AUDIO`(`Audio_ID`),
    FOREIGN KEY (`User_ID`) REFERENCES `REGISTERED_USER`(`User_ID`)
) ENGINE=InnoDB;

-- Indexes
CREATE INDEX idx_query_user ON `QUERY`(`User_ID`);
CREATE INDEX idx_query_timestamp ON `QUERY`(`Timestamp`);
CREATE INDEX idx_guest_session ON `GUEST`(`Session_ID`);
CREATE INDEX idx_registered_email ON `REGISTERED_USER`(`Email`);