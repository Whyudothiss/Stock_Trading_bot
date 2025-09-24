import os
import logging
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from pymongo import MongoClient

# Logging set Up
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
BOT_Token = os.getenv("BOT_TOKEN")

if not MONGO_URI or not BOT_Token:
    raise ValueError("Please set MONGO_URI and BOT_TOKEN environment variables")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client.portfoloio_tracker

# Collections
users_collection = db.users
transactions_collection = db.transactions

(TICKER, QUANTITY, PRICE, TRIM_TYPE, TRIM_VALUE, 
 DELETE_CONFIRM, EDIT_SELECT, EDIT_FIELD, EDIT_VALUE) = range(9)

class PortfolioTracker():
    def __init__(self):
        self.user_data = {}
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effetive_user.first_name

        # Ensure user is in the data dictionary
        if not users_collection.find_one({"user_id": user_id}):
            users_collection.insert_one({
                "user_id": user_id,
                "username": username,
                "create_at": datetime.now()
            })

        welcome_msg = """ 🎯 Welcome to myPortfolioTracker! 🎯

Your personal stock portfolio management bot. Here are the available commands:

📊 Portfolio Commands:
/buy - Record a stock purchase
/sell - Record a stock sale  
/trim - Partial sell (by % or quantity)
/current - View current holdings
/past - View sold positions with P&L
/pnl - Overall profit/loss summary
/portfolio - Quick portfolio overview

🛠 Management Commands:
/delete - Delete a transaction
/edit - Edit a transaction
/help - Show this help message

Ready to track your investments! Use /buy to record your first purchase."""


        await update.message.reply_text(welcome_msg)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.start(update, context)


    # Buy transaction flow
    async def buy_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start buy trasnsaction"""
        context.user_data['action'] = 'buy'
        await update.message.reply_text("Let's record a stock purchase!\n\n Whats the ticker symbol (e.g., AAPL):")
        return TICKER
    
    async def get_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Get ticker symbol"""
        ticker = update.message.text.strip().upper()
        context.user_date['ticker'] = ticker

        await update.message.reply_text(f"Stock: {ticker}\n\n How many shares did you buy?")
        return QUANTITY
    
    async def get_quantity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            quantity = float(update.message.text)

            if quantity <= 0:
                await update.message.reply_text("Please enter a positive number")
                return QUANTITY

            context.user_data['quantity'] = quantity
            ticker = context.user_data['ticker']
            action = context.user_data['action']

            await update.message.reply_text(f"Stock: {ticker}\nQuantity: {quantity} shares\n\n What price did you {'buy' if action == 'buy' else 'sell'} at? (USD)")
            return PRICE
        
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return QUANTITY
            
    async def get_price_and_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            price = float(update.message.text)    
            if price <= 0:
                await update.message.reply_text("PLease enter a positive price")
                return PRICE

            user_id = update.effective_user.id
            ticker = context.user_data['ticker']
            quantity = context.user_data['quantity']
            action = context.user_data['action']
            total_value = quantity * price

            transaction ={
                "user_id": user_id,
                "ticker": ticker,
                "action": action,
                "price": price,
                "quantity": quantity,
                "total_value": total_value,
                "timestamp": datetime.now()
            }

            transactions_collection.insert_one(transaction)
            await update.message.reply_text(f"✅ {ticker} purchase recorded!")
            return ConversationHandler.END
        except ValueError:
            await update.message.reoly_text("Please enter a valid price)")
            return PRICE

    async def sell_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start sell transaction"""
        context.user_data['action'] = 'sell'
        await update.message.reply_text("📉 Let's record a stock sale!\n\nWhat's the ticker symbol?")
        return TICKER
    
    def get_current_positions(self, user_id, ticker):
        """Get current positions for a user and ticker"""
        pipeline = [
            {"$match": {"user_id": user_id, "ticker": ticker}},
            {"$group": {
                "_id": "$ticker",
                "total_bought": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$quantity", 0]}}
            }}
        ]

        result = list(transactions_collection.aggregate(pipeline))
        if result:
            return result[0]["total_bought"] - result[0]("total_sold")
        return 0
    
    def get_user_holdings(self, user_id):
        """Get all current holdings for a user"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$ticker",
                "total_bought": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity:", 0]}},
                "total_sold": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$quantity", 0]}}
            }}
        ]

        results = list(transactions_collection.aggregate(pipeline))
        holdings = {}
        for result in results:
            current_qty = result["total_bought"] - result["total_sold"]
            if current_qty > 0:
                holdings[result["_id"]] = current_qty
        
        return holdings
    
    async def current_holdings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current holdings"""
        user_id = update.effective_user.id
        holdings = self.get_user_holdings(user_id)

        if not holdings:
            await update.message.reply_text("📊 You don't have any current holdings.")
            return
        
        message = "📊 Your Current Holdings:\n\n"
        for ticker, data in holdings.item():
            message += f"🏷 {ticker}: {data['quantity']} shares\n"
        
        await update.message.reply_text(message)
        

    